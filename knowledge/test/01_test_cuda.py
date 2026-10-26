import torch
print(f"{torch.cuda.is_available()}")
print(f"{torch.cuda.get_device_name()}")