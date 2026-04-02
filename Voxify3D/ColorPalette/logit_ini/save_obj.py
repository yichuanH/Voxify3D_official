import numpy as np
import argparse
from scipy.stats import rankdata
import matplotlib.pyplot as plt



def quantize_rgb_to_palette(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    將每個 voxel RGB 對應到 palette 中最接近的顏色。

    Args:
        rgb: [D, H, W, 3] 的原始 RGB voxel grid，值域為 [0, 1]
        palette: [8, 3] 的調色盤 (每列是一個 RGB)

    Returns:
        quantized_rgb: [D, H, W, 3]，每個 voxel 替換成 palette 中最接近的顏色
    """
    D, H, W, _ = rgb.shape
    flat_rgb = rgb.reshape(-1, 3)         # [N, 3]
    
    # 計算每個 voxel RGB 和 palette 中每個顏色的距離 [N, 8]
    dists = np.linalg.norm(flat_rgb[:, None, :] - palette[None, :, :], axis=2)
    nearest_idx = np.argmin(dists, axis=1)  # [N]

    # 把 index 對應到 palette 顏色
    quantized_flat = palette[nearest_idx]  # [N, 3]
    quantized_rgb = quantized_flat.reshape(D, H, W, 3)

    return quantized_rgb




def read_voxel_data(filename_alpha, filename_color):
    """讀取並加載體素資料，獲取 alpha 和 rgb 值"""
    data_alpha = np.load(filename_alpha)
    data_color = np.load(filename_color)
    alpha = data_alpha['alpha']
    rgb = data_color['color_grid']   ### max 1.00 ##min 0.04
    rgb = np.moveaxis(rgb, 0, -1)  # 把 channel 軸從第 0 軸移到最後
    #rgb = data_alpha['rgb']

    #breakpoint()

    #breakpoint()
    return alpha, rgb

def has_adjacent_voxel(alpha, i, j, k, threshold):
    """檢查體素 (i, j, k) 是否至少有一個鄰居"""
    shape = alpha.shape
    neighbors = [
        (i - 1, j, k), (i + 1, j, k),  # 左右
        (i, j - 1, k), (i, j + 1, k),  # 上下
        (i, j, k - 1), (i, j, k + 1)   # 前後
    ]
    
    for ni, nj, nk in neighbors:
        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:  # 確保索引合法
            if alpha[ni, nj, nk] >= threshold:
                return True  # 至少有一個鄰居
    return False  # 完全孤立


def save_voxel_as_obj(filename, alpha, rgb, threshold=0.85, cube_size=1.0):
    """儲存 alpha > threshold 的體素為 OBJ 格式，並排除完全孤立的體素"""
    with open(filename, 'w') as file:
        file.write("# OBJ file\n")

        # 每個體素的正方體頂點偏移量
        cube_vertices = [
            (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)
        ]
        # 每個面的頂點索引
        cube_faces = [
            (1, 2, 3, 4), (5, 6, 7, 8), (1, 5, 8, 4),
            (2, 6, 7, 3), (1, 2, 6, 5), (4, 3, 7, 8)
        ]

        vertex_count = 1  # 頂點計數

        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                for k in range(alpha.shape[2]):
                    if alpha[i, j, k] >= threshold :  #and has_adjacent_voxel(alpha, i, j, k, threshold):
                        # 取得顏色資訊
                        r, g, b = rgb[i, j, k]

                        # 記錄頂點
                        for dx, dy, dz in cube_vertices:
                            x = i + dx * cube_size
                            y = j + dy * cube_size
                            z = k + dz * cube_size
                            file.write(f"v {x} {y} {z} {r} {g} {b}\n")

                        # 記錄面
                        for face in cube_faces:
                            v1, v2, v3, v4 = (vertex_count + idx - 1 for idx in face)
                            file.write(f"f {v1} {v2} {v3} {v4}\n")

                        # 每個正方體有 8 個頂點
                        vertex_count += 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and process a .npz file.')
    parser.add_argument('--input_alpha', type=str, help='Path to the .npz file')
    parser.add_argument('--input_rgb', type=str, help='Path to the .npz file')
    parser.add_argument('--color_file', type=str, help='Path to the color palette file (.npy or .npz)')
    parser.add_argument('--output_file', type=str, help='Path to the .obj file')
    args = parser.parse_args()

    input_file_alpha = args.input_alpha
    input_file_rgb = args.input_rgb
    output_file = args.output_file
    color_file = args.color_file

    """
    palette = np.array([[1.0003059 , 0.99960226, 0.99941957],
       [0.48833302, 0.22654969, 0.0470084 ],
       [0.5705993 , 0.54224753, 0.5164246 ],
       [0.32682043, 0.23199111, 0.13431162],
       [0.4757275 , 0.39880717, 0.29684544],
       [0.55575025, 0.28577906, 0.12643838],
       [0.6769539 , 0.67797565, 0.6708498 ],
       [0.18450296, 0.13457704, 0.06722158]])
    """
      # 讀取 color palette
    if color_file.endswith(".npy"):
        palette = np.load(color_file)
    elif color_file.endswith(".npz"):
        palette_data = np.load(color_file)
        if "color_palette" in palette_data:
            palette = palette_data["color_palette"]
        else:
            raise ValueError("color_palette not found in the .npz file")
    else:
        raise ValueError("Unsupported color file format. Use .npy or .npz")

    print(f"✅ 調色盤載入完成 shape = {palette.shape}")
    
    
    # 讀取資料
    alpha, rgb = read_voxel_data(input_file_alpha, input_file_rgb)
    rgb_quantized = quantize_rgb_to_palette(rgb, palette)

    # 儲存為 .obj 格式
    save_voxel_as_obj(output_file, alpha, rgb_quantized)
