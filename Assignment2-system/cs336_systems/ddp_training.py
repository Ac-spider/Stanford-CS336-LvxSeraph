import torch
import torch.nn as nn
import torch.distributed as dist

# ─────────────────────────────────────────
# DDP 实现1：逐参数异步 AllReduce
# Purpose: 每个参数梯度就绪后立即发起异步 AllReduce，实现通信-计算 overlap
# Key concept:
#   - register_post_accumulate_grad_hook：在某个参数的 .backward() 梯度累积完毕后
#     立即回调，无需等待整个反向传播完成，从而与后续层的反向计算并行通信
#   - 初始化时 broadcast(src=0) 确保所有进程参数相同（消除随机种子差异）
#   - AllReduce SUM + 除以 world_size = AllReduce MEAN，等价于梯度平均
# ─────────────────────────────────────────
class DDPIndividualParameters(nn.Module):


   def __init__(self,module:nn.Module):
        super().__init__()

        self.module = module
        self.is_initialized = dist.is_available() and dist.is_initialized()
        self.communication_handle = []  # 存储未完成的异步通信句柄
        self.word_size = dist.get_world_size() if self.is_initialized else 1

        if self.is_initialized:
            for params in self.module.parameters():
                # 广播参数：确保所有 rank 从相同初始权重开始训练
                dist.broadcast(params.data,src=0)

                if params.requires_grad:
                    # 注册钩子：该参数梯度累积完毕后立即触发异步 AllReduce
                    params.register_post_accumulate_grad_hook(self._make_hook(params))


   def _make_hook(self,params):
        # 闭包工厂：为每个参数创建独立的钩子函数
        def hook(params):

            if not self.is_initialized:
                return

            # 先除以 world_size，再 SUM AllReduce，等价于对梯度求均值
            params.grad.data.div_(self.word_size)

            # 异步发起 AllReduce，立即返回句柄（不阻塞后续反向计算）
            handle = dist.all_reduce(params.grad.data,op=dist.ReduceOp.SUM,async_op=True)
            self.communication_handle.append(handle)

        return hook


   def forward(self,*args,**kwargs):
       # 直接委托给被包装的模块
       return self.module(*args,**kwargs)


   def finish_gradient_synchronization(self):
        # 在优化器 step() 前调用：等待所有异步通信完成
        for handle in self.communication_handle:
            handle.wait()

        self.communication_handle.clear()


# ─────────────────────────────────────────
# DDP 实现2：桶式分组 AllReduce（Bucketed DDP）
# Purpose: 将多个参数梯度拼平为大 tensor 后再 AllReduce，减少通信调用次数
# Key concept:
#   - 将参数按顺序分组到若干 bucket（每个 bucket 不超过 bucket_size_mb）
#   - 每个 bucket 内所有参数梯度就绪后，才发起一次 AllReduce
#   - 减少小 tensor 通信的 kernel launch 开销，提升通信带宽利用率
#   - torch._utils._flatten_dense_tensors：将多个 tensor 拼为一个连续 flat tensor
#   - torch._utils._unflatten_dense_tensors：将 flat tensor 拆回原始形状列表
# ─────────────────────────────────────────
class DDPBucketed(nn.Module):
    def __init__(self,model,bucket_size_mb):
        super().__init__()

        self.model = model
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024  # MB 转字节
        self.is_initialized = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_initialized else 1

        self.handles = []           # 异步通信句柄列表：(handle, bucket_id, flat_grad)
        self.buckets = []           # 每个 bucket 包含的参数列表
        self.ready_buckets = []     # 每个 bucket 中已就绪（梯度计算完）的参数计数
        self.total_buckets = []     # 每个 bucket 的总参数数

        if self.is_initialized:
            # 广播参数：确保所有 rank 初始状态相同
            for p in self.model.parameters():
                dist.broadcast(p.data,src=0)

            self._build_buckets()

    def _build_buckets(self):
        # 按参数顺序贪心分桶：超过 bucket_size 时新建桶
        current_buckets = []
        current_size = 0

        params = [p for p in self.model.parameters() if p.requires_grad]

        for param in params:
            param_size = param.numel() * param.element_size()  # 字节数

            # 当前桶已满且非空时，保存当前桶并开新桶
            if current_size + param_size > self.bucket_size_bytes and len(current_buckets) > 0:
                self.buckets.append(current_buckets)
                current_buckets = []
                current_size = 0

            current_buckets.append(param)
            current_size += param_size

        # 最后一个桶
        if len(current_buckets) > 0:
            self.buckets.append(current_buckets)

        # 初始化每个桶的就绪计数和总参数数
        self.ready_buckets = [0] * len(self.buckets)
        self.total_buckets = [len(b) for b in self.buckets]

        # 为每个桶中的每个参数注册钩子
        for bucket_id,bucket in enumerate(self.buckets):
            for param in bucket:
                param.register_post_accumulate_grad_hook(self._make_hook(param, bucket_id))

    def _make_hook(self,param,bucket_id):

        def hook(param):
            # 增加该桶的就绪计数
            self.ready_buckets[bucket_id] += 1
            # 只有桶内所有参数梯度都就绪时，才发起 AllReduce
            if self.ready_buckets[bucket_id] == self.total_buckets[bucket_id]:
                grad = [p.grad for p in self.buckets[bucket_id]]
                # 将多个梯度 tensor 拼为一个连续 flat tensor（减少通信次数）
                flat_grad = torch._utils._flatten_dense_tensors(grad)

                flat_grad.div_(self.world_size)  # 梯度平均
                # 对整个桶的 flat_grad 发起一次异步 AllReduce
                handle = dist.all_reduce(flat_grad,op=dist.ReduceOp.SUM,async_op=True)
                self.handles.append((handle,bucket_id,flat_grad))

        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # 等待所有桶的 AllReduce 完成，并将 flat_grad 拆回原始形状写回各参数梯度
        for handle,bucket_id,flat_grad in self.handles:
            handle.wait()

            grad = [p.grad for p in self.buckets[bucket_id]]
            # 将 flat tensor 拆回与原始梯度相同形状的列表
            unflat_grad = torch._utils._unflatten_dense_tensors(flat_grad,grad)

            # 将同步后的梯度写回各参数
            for orig_grad,new_grad in zip(grad,unflat_grad):
                orig_grad.copy_(new_grad)

        self.handles.clear()

    def reset_buckets(self):
        # 每个 step 结束后重置就绪计数，准备下一次反向传播
        for i in range(len(self.ready_buckets)):
            self.ready_buckets[i] = 0
