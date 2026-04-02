import torch
import torch.nn.functional as F
from typing import List
import os
import numpy as np
import matplotlib.pyplot as plt

def save_palette_color_grid(logit_grid: torch.Tensor, palette: List[List[float]], save_path: str):
    
    """
    將 logit grid 中每顆 voxel 對應的最大 logit index 轉成 RGB 值並存成 .npz。

    Args:
        logit_grid: [1, 8, D, H, W] 的 logit tensor
        palette: list of 8 RGB 色彩，每個是 [r, g, b] in [0, 1]
        save_path: 儲存路徑，例如 "ColorPalette/logit_ini/color_grid.npz"
    """
    assert logit_grid.shape[1] == len(palette), "Logit channel 與 palette 長度不一致"

    # 取得最大 logit 對應的 palette index
    with torch.no_grad():
        max_indices = torch.argmax(logit_grid, dim=1).squeeze(0)  # [D, H, W]

    D, H, W = max_indices.shape
    color_grid = torch.zeros((3, D, H, W), dtype=torch.float32)

    palette_tensor = torch.tensor(palette, dtype=torch.float32)  # [8, 3]

    # 將每個 voxel index 對應的 RGB 色彩填入 color_grid
    for idx in range(len(palette)):
        mask = (max_indices == idx)
        for c in range(3):  # R, G, B
            color_grid[c][mask] = palette_tensor[idx][c]

    # 轉成 numpy 並存成 .npz
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, color_grid=color_grid.cpu().numpy())

    print(f"✅ Saved color palette grid to: {save_path}")


def convert_densegrid_to_logitgrid(densegrid, palette: List[List[float]], temperature: float = 0.5):
    """
    將 DenseGrid (RGB channels) 轉換為 LogitGrid (palette-based logits)。

    Args:
        densegrid: torch.Tensor of shape [1, 3, D, H, W]
        palette: list of 8 RGB colors, each RGB is a list of 3 floats in [0, 1]
        temperature: softmax temperature (a)

    Returns:
        logit_grid: torch.Tensor of shape [1, 8, D, H, W]
    """
    assert densegrid.shape[1] == 3, "Input must be a 3-channel DenseGrid"

    device = densegrid.device
    B, C, D, H, W = densegrid.shape
    
    densegrid = torch.sigmoid(densegrid)  
    # flatten to [N, 3]
    rgb = densegrid[0].reshape(3, -1).T  # [N, 3]

    # rgb.max() = 56.3902
    # rgb.min() = (-31.1640
    # palette: [8, 3]
    palette_tensor = torch.tensor(palette, dtype=torch.float32, device=device)
    # palette_tensor.max() = 1.0003
    # palette_tensor.min() = 0.0470
    # compute distance: [N, 8]
    
    dists = torch.cdist(rgb.unsqueeze(0), palette_tensor.unsqueeze(0)).squeeze(0)

    # 將距離轉換為偏好分數（logits）並用 softmax 平滑處理
    logits = -dists / temperature
    #probs = F.softmax(logits, dim=1)  # [N, 8]
    # reshape back to [1, 8, D, H, W]
    logit_grid = logits.T.reshape(1, len(palette), D, H, W)
    
    
    print("🔥 max logit index count:", torch.bincount(torch.argmax(logit_grid, dim=1).flatten()))
    save_palette_color_grid(
    logit_grid=logit_grid,
    palette=palette,
    save_path="ColorPalette/logit_ini/color_grid.npz"
    )
    
    #breakpoint()
    return logit_grid


#import torch

def random_densegrid_to_logitgrid(densegrid, palette, temperature=0.5):
    """
    隨機初始化每個 voxel 的 palette logits，形成 logit grid。

    Args:
        world_size: tuple (D, H, W)
        palette: list of 8 RGB 顏色，每個是 [r,g,b]，只用來記錄 logit channel 數
        temperature: 未使用，但保留參數形式一致
        mean, std: 用來調整 logits 的初始化分佈

    Returns:
        logit_grid: torch.Tensor of shape [1, 8, D, H, W]
    """
    C = len(palette)
    _, _, D, H, W = densegrid.shape
    
    mean=0.0
    std=0.01
    
    logit_grid = torch.randn(1, C, D, H, W) * std + mean  # shape: [1, 8, D, H, W]
    
    save_palette_color_grid(
    logit_grid=logit_grid,
    palette=palette,
    save_path="ColorPalette/logit_ini/random_grid.npz"
    )
    
    
    return logit_grid
