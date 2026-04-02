import numpy as np
from scipy.ndimage import label

def process_uncertainty_map(uncertainty_maps, cell_size):

    threshold_pixels = (cell_size * cell_size) // 2  # Minimum pixel count for connected components

    # Validate input shape
    if uncertainty_maps.shape != (6, 1200, 1200, 1):
        raise ValueError("Input array must have shape (6, 1200, 1200).")

    processed_maps = []

    for i in range(uncertainty_maps.shape[0]):
        # 取得單張 map
        uncertainty_mask = uncertainty_maps[i, :, :, 0]
    
        # 1. 設定閾值進行二值化處理
        #uncertainty_mask[uncertainty_mask > 0.95] = 0

          ## 原本的

        treshold = 0.016 #0.008

        uncertainty_mask[(uncertainty_mask > treshold) & (uncertainty_mask <= 0.85)] = 1
        uncertainty_mask[uncertainty_mask <= treshold] = 0
        
        
        #無差別上色
        #uncertainty_mask[(uncertainty_mask >= 0)] = 1

        # 2. 移除小區塊 (小於閾值的連通區域)
        labeled_array, num_features = label(uncertainty_mask)  # 標記連通區域

        for region_label in range(1, num_features + 1):
            region_size = np.sum(labeled_array == region_label)  # 計算連通區域大小
            if region_size < threshold_pixels:
                uncertainty_mask[labeled_array == region_label] = 0  # 移除小區域

        # 加入處理後的結果
        processed_maps.append(uncertainty_mask)

    # 堆疊處理後的所有 map
    return np.stack(processed_maps, axis=0)[..., np.newaxis]

