import torch

def log_cuda_mem(tag):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[CUDA] {tag:20s} allocated={alloc:.2f} GiB, reserved={reserved:.2f} GiB")