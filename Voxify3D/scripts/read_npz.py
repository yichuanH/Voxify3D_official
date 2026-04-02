import os
import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_depth_histogram(scene, save_path):
    # 檔案路徑
    file_path = f"../logs/our_data/{scene}/dvgo_adventurer_ortho/render_test_fine_last/test_depths.npz"

    # 確認檔案是否存在
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # 讀取 .npz 檔案
    try:
        data = np.load(file_path)
        print(f"Keys in the file: {list(data.keys())}")

        # 假設只有一個 key：depths
        depths = data['depths'].flatten()

        # 繪製直方圖
        plt.figure(figsize=(10, 6))
        plt.hist(depths, bins=50, range=(depths.min(), depths.max()), color='blue', alpha=0.7)
        plt.title("Histogram of Depth Values")
        plt.xlabel("Depth Value")
        plt.ylabel("Frequency")
        plt.grid(True)

        # 確保保存目錄存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Histogram saved at {save_path}")
        plt.close()
        breakpoint()
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # 替換為實際場景名稱和保存路徑
    scene = "adventurer"
    save_path = "./depth_histogram.png"

    plot_and_save_depth_histogram(scene, save_path)
