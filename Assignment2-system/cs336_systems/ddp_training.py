import torch
import torch.nn as nn
import torch.distributed as dist

class DDPIndividualParameters(nn.Module):


   def __init__(self,module:nn.Module):
        super().__init__()

        self.module = module
        self.is_initialized = dist.is_available() and dist.is_initialized()
        self.communication_handle = []
        self.word_size = dist.get_world_size() if self.is_initialized else 1

        if self.is_initialized:
            for params in self.module.parameters():

                dist.broadcast(params.data,src=0)

                if params.requires_grad:
                    params.register_post_accumulate_grad_hook(self._make_hook(params))


   def _make_hook(self,params):

        def hook(params):

            if not self.is_initialized:
                return

            params.grad.data.div_(self.word_size)

            handle = dist.all_reduce(params.grad.data,op=dist.ReduceOp.SUM,async_op=True)
            self.communication_handle.append(handle)

        return hook


   def forward(self,*args,**kwargs):
       return self.module(*args,**kwargs)


   def finish_gradient_synchronization(self):

       for handle in self.communication_handle:
            handle.wait()

       self.communication_handle.clear()


class DDPBucketed(nn.Module):
    def __init__(self,model,bucket_size_mb):
        super().__init__()

        self.model = model
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.is_initialized = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_initialized else 1

        self.handles = []
        self.buckets = []
        self.ready_buckets = []
        self.total_buckets = []

        if self.is_initialized:
            for p in self.model.parameters():
                dist.broadcast(p.data,src=0)

            self._build_buckets()

    def _build_buckets(self):
        current_buckets = []
        current_size = 0

        params = [p for p in self.model.parameters() if p.requires_grad]

        for param in params:
            param_size = param.numel() * param.element_size()

            if current_size + param_size > self.bucket_size_bytes and len(current_buckets) > 0:
                self.buckets.append(current_buckets)
                current_buckets = []
                current_size = 0

            current_buckets.append(param)
            current_size += param_size

        if len(current_buckets) > 0:
            self.buckets.append(current_buckets)

        self.ready_buckets = [0] * len(self.buckets)
        self.total_buckets = [len(b) for b in self.buckets]

        for bucket_id,bucket in enumerate(self.buckets):
            for param in bucket:
                param.register_post_accumulate_grad_hook(self._make_hook(param, bucket_id))

    def _make_hook(self,param,bucket_id):

        def hook(param):
            self.ready_buckets[bucket_id] += 1
            if self.ready_buckets[bucket_id] == self.total_buckets[bucket_id]:
                grad = [p.grad for p in self.buckets[bucket_id]]
                flat_grad = torch._utils._flatten_dense_tensors(grad)

                flat_grad.div_(self.world_size)
                handle = dist.all_reduce(flat_grad,op=dist.ReduceOp.SUM,async_op=True)
                self.handles.append((handle,bucket_id,flat_grad))

        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def finish_gradient_synchronization(self):

        for handle,bucket_id,flat_grad in self.handles:
            handle.wait()

            grad = [p.grad for p in self.buckets[bucket_id]]

            unflat_grad = torch._utils._unflatten_dense_tensors(flat_grad,grad)

            for orig_grad,new_grad in zip(grad,unflat_grad):
                orig_grad.copy_(new_grad)

        self.handles.clear()

    def reset_buckets(self):
        for i in range(len(self.ready_buckets)):
            self.ready_buckets[i] = 0















