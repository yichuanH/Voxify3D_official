import os
import numpy as np
import matplotlib.pyplot as plt

def save_depth_data(render_depth, depth_target, output_dir):
    """
    Save render_depth and depth_target as separate .npz files and generate visualizations.

    Args:
        render_depth (np.ndarray): Rendered depth values.
        depth_target (np.ndarray): Ground truth depth values.
        output_dir (str): Directory to save the .npz files and plots.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save render_depth
    render_depth_path = os.path.join(output_dir, "render_depth.npz")
    np.savez_compressed(render_depth_path, render_depth=render_depth)
    print(f"Render depth saved at: {render_depth_path}")

    # Save depth_target
    depth_target_path = os.path.join(output_dir, "depth_target.npz")
    np.savez_compressed(depth_target_path, depth_target=depth_target)
    print(f"Depth target saved at: {depth_target_path}")

    # Plot histogram
    plt.hist(render_depth.flatten(), bins=50, alpha=0.5, label="Render Depth")
    plt.hist(depth_target.flatten(), bins=50, alpha=0.5, label="Depth Target")
    plt.xlabel("Depth Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram of Depth Distributions")
    histogram_path = os.path.join(output_dir, "depth_histogram.png")
    plt.savefig(histogram_path)
    plt.close()
    print(f"Histogram saved at: {histogram_path}")

    # Scatter plot
    plt.scatter(depth_target.flatten(), render_depth.flatten(), alpha=0.5, s=1)
    plt.xlabel("Depth Target")
    plt.ylabel("Render Depth")
    plt.title("Scatter Plot of Depth Target vs. Render Depth")
    plt.plot([depth_target.min(), depth_target.max()],
             [depth_target.min(), depth_target.max()],
             color='red', linestyle='--', label='Ideal Line')
    plt.legend()
    scatter_path = os.path.join(output_dir, "depth_scatter_plot.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"Scatter plot saved at: {scatter_path}")

if __name__ == "__main__":
    # 指定文件路徑
    depth_target_path = "depth_target_qqqq.npz"
    render_depth_path = "render_depth_qqqq.npz"
    
    # 從文件讀取數據
    depth_target = np.load(depth_target_path)['depth_target']
    render_depth = np.load(render_depth_path)['render_depth']
    
    # 指定輸出目錄
    output_dir = "./depth_data"
    
    # 保存及可視化
    save_depth_data(render_depth, depth_target, output_dir)
