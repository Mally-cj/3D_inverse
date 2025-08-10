import torch

# 显式释放 GPU 缓存
torch.cuda.empty_cache()

# 如果需要，可以强制清除缓存
torch.cuda.ipc_collect()