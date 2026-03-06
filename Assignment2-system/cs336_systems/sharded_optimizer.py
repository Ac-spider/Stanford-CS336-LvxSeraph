from typing import Type, Any, Callable
import torch
import torch.distributed as dist

class ShardedOptimizer(torch.optim.Optimizer):

    def __init__(self,params,optimizer_cls,**kwargs):

        self.global_index = 0
        self.param_to_rank = {}
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        defaults = kwargs.copy()
        super().__init__(params,defaults)

        inner_params = []
        for group in self.param_groups:
            sharded_params = [p for p in group['params'] if self.param_to_rank[p] == self.rank]

            if len(sharded_params) > 0:
                inner_param = {**group,'params':sharded_params}
                inner_params.append(inner_param)

        self.inner_optimizer = optimizer_cls(inner_params)


    def add_param_group(self, param_group: dict[str, Any]) -> None:

        if self.world_size>1:
            for p in param_group['params']:
                self.param_to_rank[p] = self.global_index % self.world_size
                self.global_index += 1

        super().add_param_group(param_group)

        if hasattr(self,'inner_optimizer') and getattr(self,'inner_optimizer',None) is not None:
            sharded_params = [p for p in param_group['params'] if self.param_to_rank[p] == self.rank]
            if len(sharded_params) > 0:
                inner_param = {**param_group, 'params': sharded_params}
                self.inner_optimizer.add_param_group(inner_param)

    def step(self, closure: Callable[[], float] | None = None,**kwargs) -> float | None:
        loss =None

        if len(self.inner_optimizer.param_groups) > 0:
            loss = self.inner_optimizer.step(closure)

        if self.world_size > 1:
            for group in self.param_groups:
                for p in group['params']:
                    owner_rank = self.param_to_rank[p]
                    dist.broadcast(p.data,owner_rank)

        return loss














