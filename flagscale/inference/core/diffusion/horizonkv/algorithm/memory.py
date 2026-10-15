# Modified for the HorizonKV inference artifact; see NOTICE.
import torch

gpu = torch.device("cuda")


def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu
    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats["active_bytes.all.current"]
    bytes_reserved = memory_stats["reserved_bytes.all.current"]
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / 1024**3


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(
        f"Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB"
    )
    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
            torch.cuda.empty_cache()
            return
        if hasattr(m, "weight"):
            m.to(device=target_device)
    model.to(device=target_device)
    torch.cuda.empty_cache()
    return
