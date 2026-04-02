import torch

# 指定檔案路徑
ckpt_path = '../logs/our_data/adventurer/dvgo_adventurer_ortho/coarse_last.tar'
ckpt = torch.load(ckpt_path, map_location='cpu')

# 檢查是否有 uncertainty grid
print("Uncertainty grid keys in coarse checkpoint:")
for key in ckpt['model_state_dict']:
    if "uncertainty_grid" in key:
        print(key)