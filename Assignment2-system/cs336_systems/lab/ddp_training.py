import torch
import torch.nn as nn
import torch.distributed as dist


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        """
        基础版 DDP 容器。
        核心逻辑：每个参数在反向传播计算完梯度后，立刻独立发起 All-Reduce 通信。
        这种方式通信极其碎片化，对于包含大量小参数的模型，网络延迟（Latency）会成为训练瓶颈。
        """
        super().__init__()
        self.module = module
        # 用于存储异步通信的句柄（handle），以便在反向传播结束后统一等待通信完成
        self.communication_handles = []

        # 检查分布式环境是否已初始化（例如是否调用了 dist.init_process_group）
        self.is_initialized = dist.is_available() and dist.is_initialized()
        # 获取当前分布式环境中的总进程数（即 GPU 数量），单机单卡时默认为 1
        self.world_size = dist.get_world_size() if self.is_initialized else 1

        if self.is_initialized:
            for param in self.module.parameters():
                # 状态同步：在训练开始前，将 rank 0（主节点）的初始权重强行广播给所有 rank
                # 这是为了确保所有 GPU 上的模型在 step 0 时拥有完全一致的初始参数
                dist.broadcast(param.data, src=0)

                # 为每个需要求导的参数注册“梯度累加完成后的回调钩子”（Hook）
                # 当 loss.backward() 执行到该参数，且该参数的梯度计算并累加完毕时，会自动触发 _make_hook 返回的函数
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param):
        """
        闭包函数：为特定参数生成专属的反向传播钩子。
        """

        def hook(param):
            if not self.is_initialized:
                return

            # 梯度求平均：分布式训练中，通常需要将所有 GPU 上的梯度相加后除以 GPU 数量
            param.grad.data.div_(self.world_size)

            # 异步执行 All-Reduce 通信
            # op=dist.ReduceOp.SUM 表示将所有 GPU 上的该参数梯度进行求和
            # async_op=True 表示这是一个非阻塞操作，CPU 发出指令后立即返回，GPU 在后台进行网络通信
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)

            # 保存通信句柄，后续需要调用 handle.wait() 确保数据传输真正完成
            self.communication_handles.append(handle)

        return hook

    def forward(self, *args, **kwargs):
        # 前向传播直接调用底层模块
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        同步屏障。
        必须在 optimizer.step() 之前调用，确保所有后台异步传输的梯度都已经到达本地 GPU。
        """
        for handle in self.communication_handles:
            handle.wait()  # 阻塞当前进程，直到该 handle 对应的 All-Reduce 完成
        self.communication_handles.clear()  # 清空句柄，为下一个 step 做准备


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        """
        进阶版 DDP 容器（带有梯度分桶策略）。
        核心逻辑：不再是“计算完一个参数就发一次通信”，而是将多个参数的梯度打包进一个“桶（Bucket）”里。
        当桶装满（或者桶内所有参数梯度都就绪）时，才发起一次整块内存的 All-Reduce 通信。
        优势：大幅减少网络通信的次数（降低 Latency 成本），提升带宽高利用率。
        """
        super().__init__()
        self.module = module
        # 将 MB 转换为 Bytes (1 MB = 1024 * 1024 Bytes)
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024

        self.is_initialized = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_initialized else 1

        self.buckets = []  # 列表的列表，存储每个桶包含的参数引用（按组划分）
        self.ready_counts = []  # 状态计数器：记录每个桶当前已经计算完梯度的参数数量
        self.total_counts = []  # 静态信息：记录每个桶总共包含多少个参数
        self.handles = []  # 记录异步通信的 handle、对应的桶 ID 以及展平后的梯度张量

        if self.is_initialized:
            # 同样需要在初始化时对齐所有进程的模型参数
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        # 构建分桶结构
        self._build_buckets()

    def _build_buckets(self):
        # 反向传播的计算顺序通常是从网络的最后一层到第一层。
        # 为了让通信与计算完美重叠，我们将参数【逆序】装入桶中。
        # 这样网络后面的层计算完后，可以最先凑满一个桶并触发通信，此时 GPU 可以继续计算前面层的梯度。
        params = [p for p in self.module.parameters() if p.requires_grad]
        params.reverse()

        current_bucket = []
        current_size = 0

        # 贪心算法：按顺序把参数塞进桶里，超过容量限制就换下一个新桶
        for p in params:
            # 计算当前参数占用的字节数 = 元素总数 * 单个元素的字节大小（如 float32 占 4 bytes）
            p_size = p.numel() * p.element_size()

            # 如果加上当前参数会超出预设的桶大小，并且当前桶不是空的，则封桶并开辟新桶
            if current_size + p_size > self.bucket_size_bytes and len(current_bucket) > 0:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append(p)
            current_size += p_size

        # 处理循环结束后剩下的最后一个未满的桶
        if len(current_bucket) > 0:
            self.buckets.append(current_bucket)

        # 初始化每个桶的计数器
        self.ready_counts = [0] * len(self.buckets)
        self.total_counts = [len(b) for b in self.buckets]

        # 遍历生成的所有桶，为其中的每个参数分配对应的桶 ID，并注册 Hook
        for bucket_id, bucket_params in enumerate(self.buckets):
            for p in bucket_params:
                # 把 bucket_id 传给 hook，这样参数计算完梯度后，就知道该去更新哪个桶的进度条
                p.register_post_accumulate_grad_hook(self._make_hook(p, bucket_id))

    def _make_hook(self, param, bucket_id):
        def hook(param):
            # 每当有个参数梯度就绪，对应桶的“已准备好”计数器加 1
            self.ready_counts[bucket_id] += 1

            # 检查触发条件：当这个桶里【所有的】参数梯度都计算完毕时，触发桶级别的批量通信
            if self.ready_counts[bucket_id] == self.total_counts[bucket_id]:
                if not self.is_initialized:
                    return

                bucket_params = self.buckets[bucket_id]
                # 收集当前桶内所有参数的梯度张量
                grads = [p.grad.data for p in bucket_params]

                # 【核心操作】：将物理内存上离散的多个梯度张量，强制拼接成一整块连续的 1D 内存
                # 底层通信库（如 NCCL/GloO）对连续大块内存的传输效率远高于多个小碎块
                flat_grad = torch._utils._flatten_dense_tensors(grads)
                # 在通信前先除以进程数，求均值
                flat_grad.div_(self.world_size)

                # 对这块巨大的连续内存发起异步 All-Reduce
                handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)

                # 将 handle、桶 ID、以及这块连续内存保存起来
                self.handles.append((handle, bucket_id, flat_grad))

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        等待所有的桶通信完成，并将聚合后的连续内存拆解回原梯度张量中。
        """
        for handle, bucket_id, flat_grad in self.handles:
            # 确保这块大内存的 All-Reduce 通信已物理完成
            handle.wait()

            # 找到对应桶的原始参数的梯度引用
            bucket_params = self.buckets[bucket_id]
            grads = [p.grad.data for p in bucket_params]

            # 【核心操作】：将通信完的 1D 扁平内存，按照原来各个梯度的 shape 重新切割/拆解
            unflat_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)

            # 将拆解后（已经聚合完毕）的数据，硬拷贝回原始的模型参数梯度内存池中
            # 这样 optimizer.step() 更新权重时就能拿到最新的、平均后的梯度
            for orig_grad, new_grad in zip(grads, unflat_grads):
                orig_grad.copy_(new_grad)

        self.handles.clear()

    def reset_buckets(self):
        """
        在每个 step 开始前（或者前向传播前）重置计数器。
        如果不重置，下一个 step 反向传播时 ready_counts 将无法准确反映状态。
        """
        for i in range(len(self.ready_counts)):
            self.ready_counts[i] = 0