#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse

# =========================
# Scene-level configuration
# =========================
scene_configs = {
    "fallguy" : [[30, "kmeans_rare", 8]],
    # "redpanda" : [[50, "kmeans", 6]],
    # "angry" : [[20, "kmeans_rare", 6], [25, "kmeans_rare", 6], [30, "kmeans_rare", 6], [40, "kmeans_rare", 6], [50, "kmeans_rare", 6]],
    # "bear" :  [[20, "kmeans_rare", 6], [25, "kmeans_rare", 6], [30, "kmeans_rare", 6], [40, "kmeans_rare", 6], [50, "kmeans_rare", 6]],
    # "benz" :  [[20, "kmeans_rare", 6], [25, "kmeans_rare", 6], [30, "kmeans_rare", 6], [40, "kmeans_rare", 6], [50, "kmeans_rare", 6]],
    # "blue_horse" :  [[20, "kmeans_rare", 6], [25, "kmeans_rare", 6], [30, "kmeans_rare", 6], [40, "kmeans_rare", 6], [50, "kmeans_rare", 6]],
    #"corgi" :  [[20, "kmeans_rare", 6], [25, "kmeans_rare", 6], [30, "kmeans_rare", 6], [40, "kmeans_rare", 6], [50, "kmeans_rare", 6]],
    # "cottage" :  [[20, "kmeans_rare", 6], [25, "kmeans_rare", 6], [30, "kmeans_rare", 6], [40, "kmeans_rare", 6], [50, "kmeans_rare", 6]],
    # "dog" :  [[20, "kmeans_rare", 6], [25, "kmeans_rare", 6], [30, "kmeans_rare", 6], [40, "kmeans_rare", 6], [50, "kmeans_rare", 6]],
    # "elephant" :  [[20, "kmeans_rare", 6], [25, "kmeans_rare", 6], [30, "kmeans_rare", 6], [40, "kmeans_rare", 6], [50, "kmeans_rare", 6]],
}

def main():
    parser = argparse.ArgumentParser(description="Batch runner for execute_gumbel.py")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id(s), e.g. 0 or 0,1")
    parser.add_argument("--data_root", type=str, default="GLB", help="Root directory for the data")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["train_ortho", "train_6views", "train_o+6", "train_6order", "train_o+6+od", "train_6+od", "save"],
        help="Execution mode passed to execute_gumbel.py"
    )
    args = parser.parse_args()

    device = args.device
    data_root = args.data_root
    mode = args.mode

    # 取得原始工作目錄路徑 (absolute path)
    original_cwd = os.getcwd()
    
    # 檢查 DVGO_Gumbel 目錄是否存在
    project_subfolder = "Voxify3D"
    if not os.path.exists(project_subfolder):
        raise SystemExit(f"❌ 錯誤: 找不到專案資料夾 {project_subfolder}！")

    failed_jobs = []

    for scene, cfg_list in scene_configs.items():
        if not isinstance(cfg_list, list):
            continue

        # --- 新增步驟：進入子目錄檢查並執行 Blender ---
        print(f"\n🔍 [Step 1] 檢查 {scene} 的渲染資料...")
        
        # 切換到 DVGO_Gumbel
        os.chdir(project_subfolder)
        
        # 檢查 data/{root}/{scene}/ortho 是否存在
        # 注意：這裡路徑是相對於 DVGO_Gumbel
        check_path = os.path.join("data", data_root, scene, "ortho")
        
        if not os.path.exists(check_path):
            print(f"⚠️  找不到 ortho 資料夾，正在執行 Blender 渲染：{scene}...")
            # 這裡使用系統 Python 3.12 繞過 PEP 668 限制的環境
            # 加上 --break-system-packages 確保之前的 numpy 安裝能生效
            # blender_cmd = (
            #     f"blender -b -P glb2img.py -- "
            #     f"--input_dir data/{data_root}/{scene} "
            #     f"--n_views 100 --res 1200"
            # )
            blender_exe = "/project2/yichuanh/blender-4.0.2-linux-x64/blender"

            glb2img_script = "/project2/yichuanh/Voxify3D/DVGO_Gumbel/glb2img.py"
            blender_cmd = (
                f"{blender_exe} -b -P {glb2img_script} -- "
                f"--input_dir data/{data_root}/{scene} "
                f"--n_views 100 --res 1200"
            )
            
            try:
                subprocess.run(blender_cmd, shell=True, check=True)
                print(f"✅ Blender 渲染完成：{scene}")
            except subprocess.CalledProcessError:
                print(f"❌ Blender 渲染失敗：{scene}")
                os.chdir(original_cwd) # 失敗也要記得跳回去
                failed_jobs.append((scene, "Blender Error", ""))
                continue
        else:
            print(f"✨ {scene} 已有渲染資料，跳過渲染步驟。")

        # 跳出 DVGO_Gumbel 回到原始目錄
        os.chdir(original_cwd)

        # --- 原有步驟：執行 execute_gumbel.py ---
        for cfg in cfg_list:
            cell_size, palette, color_num = cfg

            print(f"\n🚀 [Step 2] Starting Task: scene={scene}, mode={mode}")

            command = (
                f"python execute_gumbel_color_palette.py "
                f"--scene {scene} "
                f"--cell_size {cell_size} "
                f"--data_root {data_root} "
                f"--gpu {device} "
                f"--color_num {color_num} "
                f"--palette_mode {palette} "
                # f"--mode train_ortho"
            )
            # 如果需要傳遞 mode，取消下面註解
            # if mode: command += f" --mode {mode}"

            print(f"Running:\n{command}\n")

            try:
                subprocess.run(command, shell=True, text=True, check=True)
            except subprocess.CalledProcessError:
                print(f"❌ 執行失敗: scene={scene}, cell_size={cell_size}")
                failed_jobs.append((scene, cell_size, palette))

    print("\n✅ 所有任務處理完畢！")

    if failed_jobs:
        print("\n以下任務執行失敗：")
        for s, c, p in failed_jobs:
            print(f"   Scene: {s}, Info: {c}")
    else:
        print("\n🎉 全部成功！")

if __name__ == "__main__":
    main()