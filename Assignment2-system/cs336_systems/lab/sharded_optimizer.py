from typing import Type, Any
import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    # params: 模型的参数迭代器
    # optimizer_cls: 实际使用的底层优化器类 (例如 torch.optim.AdamW)
    # **kwargs: 传递给底层优化器的超参数 (如 lr, weight_decay)
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):

        # 1. 预初始化状态变量
        # PyTorch 的 Optimizer 父类在 __init__ 内部会调用 self.add_param_group()
        # 为了防止在执行父类 __init__ 时这些变量还未定义而报错，必须提前初始化。
        self.global_index = 0  # 用于记录参数的全局序号，方便进行轮询分配
        self.param_to_rank = {}  # 核心字典：记录 [具体的参数 Tensor -> 负责更新它的 GPU Rank]

        # 获取当前分布式环境的进程总数 (world_size) 和当前进程的编号 (rank)
        # 如果没有初始化分布式环境，则默认单机单卡 (world_size=1, rank=0)
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        defaults = kwargs.copy()

        # 2. 调用父类构造器
        # 这一步会自动调用被子类重写的 self.add_param_group()
        # 此时，所有的参数都会被轮询分配给各个 Rank，并记录在 self.param_to_rank 中
        super().__init__(params, defaults)

        # 3. 构造属于当前 Rank 的“私有”参数组
        inner_groups = []
        for group in self.param_groups:
            # 过滤出“归属权”属于当前 Rank 的参数
            sharded_params = [p for p in group['params'] if self.param_to_rank.get(p) == self.rank]

            # 只有当该组分配到了参数时，才传入内部优化器
            # 如果不加这个判断，直接传入空列表，某些 PyTorch 优化器可能会抛出异常
            if len(sharded_params) > 0:
                inner_group = {**group, 'params': sharded_params}
                inner_groups.append(inner_group)

        # 4. 实例化真正的内部优化器
        # 这个 inner_optimizer 才是真正干活的（比如计算梯度、应用动量等）
        # 但它只占用当前 Rank 负责的那部分参数的显存（大大降低了优化器状态的显存开销）
        self.inner_optimizer = optimizer_cls(inner_groups, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        # 安全回退：确保在 add_param_group 被外部单独调用时，核心属性已经初始化
        if not hasattr(self, 'global_index'):
            self.global_index = 0
            self.param_to_rank = {}
            self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        # 核心分配逻辑：以轮询 (Round-robin) 的方式为每个参数分配其“归属 Rank”
        if 'params' in param_group:
            param_group['params'] = list(param_group['params'])  # 确保 params 是列表格式
            for p in param_group['params']:
                # 如果这个参数还没有被分配
                if p not in self.param_to_rank:
                    # 使用取模运算进行轮询分配。
                    # 例如有 4 张卡，参数 0 给卡0，参数 1 给卡1，参数 4 再次给卡0...
                    self.param_to_rank[p] = self.global_index % self.world_size
                    self.global_index += 1

        # 让父类接管标准的 group 注册，将其加入到 self.param_groups 中
        super().add_param_group(param_group)

        # 动态更新内部优化器：
        # 如果在训练中途动态加入新的参数组（比如进行某些特定的微调操作），需要同步更新 inner_optimizer
        if hasattr(self, 'inner_optimizer') and getattr(self, 'inner_optimizer', None) is not None:
            # 同样地，只把属于当前 Rank 的新参数挑出来给 inner_optimizer
            sharded_params = [p for p in param_group['params'] if self.param_to_rank.get(p) == self.rank]
            if len(sharded_params) > 0:
                inner_group = {**param_group, 'params': sharded_params}
                self.inner_optimizer.add_param_group(inner_group)

    def step(self, closure=None, **kwargs):
        loss = None

        # 1. 局部更新 (Local Step)
        # 当前 Rank 的内部优化器只负责更新它拥有的那部分参数
        # 此时，所有参数都已经有了对应的梯度（通常是通过 DistributedDataParallel 的 all-reduce 得到的）
        if len(self.inner_optimizer.param_groups) > 0:
            loss = self.inner_optimizer.step(closure)

        # 2. 全局同步 (Global Synchronization)
        # 核心：更新完成后，负责更新的那个 Rank 将最新的参数广播 (Broadcast) 给所有其他 Rank
        if self.world_size > 1:
            for group in self.param_groups:
                for p in group['params']:
                    # 找到这个参数的主人是谁
                    owner_rank = self.param_to_rank[p]

                    # src=owner_rank 表示从 owner_rank 发送数据，其他 rank 接收数据
                    # broadcast 操作是就地 (in-place) 修改的，这意味着其他 rank 原本旧的参数值
                    # 会被 owner_rank 传过来的最新值直接覆盖
                    dist.broadcast(p.data, src=owner_rank)

        return loss