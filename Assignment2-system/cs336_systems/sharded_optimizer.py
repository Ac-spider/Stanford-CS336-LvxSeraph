from typing import Type, Any, Callable
import torch
import torch.distributed as dist

# ─────────────────────────────────────────
# ZeRO-1 分片优化器（Sharded Optimizer）
# Purpose: 将优化器状态（m、v、参数副本）按 rank 均匀分片，
#          每个 rank 只维护 1/K 的优化器状态，显存从 O(N) 降至 O(N/K)
# Key concept（ZeRO Stage 1）：
#   - 分配策略：第 i 个参数分配给 rank = i % world_size（轮询均匀分配）
#   - 每个 rank 只对自己负责的参数调用 inner_optimizer.step()
#   - step 完成后对全量参数做 Broadcast，使所有 rank 的参数保持同步
#   - 梯度仍然全量保存（ZeRO-1 只分片优化器状态，ZeRO-2 才分片梯度）
# ─────────────────────────────────────────
class ShardedOptimizer(torch.optim.Optimizer):

    def __init__(self,params,optimizer_cls,**kwargs):
        # global_index：用于轮询分配，每添加一个参数自增1
        self.global_index = 0
        # param_to_rank：记录每个参数张量归属的 rank
        self.param_to_rank = {}
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        defaults = kwargs.copy()
        # 调用父类 __init__，会触发 add_param_group → 完成轮询分配
        super().__init__(params,defaults)

        # 从所有 param_groups 中筛选本 rank 负责的参数，构建 inner_optimizer
        inner_params = []
        for group in self.param_groups:
            # 只保留分配给当前 rank 的参数
            sharded_params = [p for p in group['params'] if self.param_to_rank[p] == self.rank]

            if len(sharded_params) > 0:
                inner_param = {**group,'params':sharded_params}
                inner_params.append(inner_param)

        # inner_optimizer：只维护本 rank 负责参数的 m、v 等优化器状态
        self.inner_optimizer = optimizer_cls(inner_params)


    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # 覆盖父类方法：在添加参数组时完成轮询分配
        if self.world_size>1:
            for p in param_group['params']:
                # 轮询分配：参数 i 分给 rank i % world_size
                self.param_to_rank[p] = self.global_index % self.world_size
                self.global_index += 1

        super().add_param_group(param_group)

        # 如果 inner_optimizer 已创建，也同步更新（动态添加参数组的场景）
        if hasattr(self,'inner_optimizer') and getattr(self,'inner_optimizer',None) is not None:
            sharded_params = [p for p in param_group['params'] if self.param_to_rank[p] == self.rank]
            if len(sharded_params) > 0:
                inner_param = {**param_group, 'params': sharded_params}
                self.inner_optimizer.add_param_group(inner_param)

    def step(self, closure: Callable[[], float] | None = None,**kwargs) -> float | None:
        loss = None

        # 只对本 rank 负责的参数执行优化器更新（AdamW m/v 状态只在对应 rank 上存在）
        if len(self.inner_optimizer.param_groups) > 0:
            loss = self.inner_optimizer.step(closure)

        # Broadcast 同步：每个参数由 owner_rank 广播到所有其他 rank
        # 确保所有 rank 的模型参数完全一致（下一次前向传播结果相同）
        if self.world_size > 1:
            for group in self.param_groups:
                for p in group['params']:
                    owner_rank = self.param_to_rank[p]
                    # owner_rank 广播更新后的参数，其他 rank 接收覆盖本地副本
                    dist.broadcast(p.data,owner_rank)

        return loss
