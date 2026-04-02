#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse

# =========================
# Scene-level configuration
# =========================
scene_configs = {
    "Chiikawa" : [[30, "kmeans_rare", 4], [30, "kmeans_rare", 3], [40, "kmeans_rare", 4] ] #"kmeans", "maxmin", "mediancut", "sa"
    # "redpanda" : [[50, "kmeans", 6]],
    # "angry" : [[20, "kmeans_rare", 6], [25, "kmeans_rare", 6], [30, "kmeans_rare", 6], [40, "kmeans_rare", 6], [50, "kmeans_rare", 6]],
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

    # Get absolute path of the original working directory
    original_cwd = os.getcwd()

    # Check if the project subfolder exists
    project_subfolder = "Voxify3D"
    if not os.path.exists(project_subfolder):
        raise SystemExit(f"Error: project folder not found: {project_subfolder}")

    failed_jobs = []

    for scene, cfg_list in scene_configs.items():
        if not isinstance(cfg_list, list):
            continue

        # --- Step 1: Check and run Blender rendering if needed ---
        print(f"\n[Step 1] Checking render data for {scene}...")

        os.chdir(project_subfolder)

        # Check if data/{root}/{scene}/ortho exists (path relative to project_subfolder)
        check_path = os.path.join("data", data_root, scene, "ortho")

        if not os.path.exists(check_path):
            print(f"ortho folder not found, running Blender render for: {scene}...")
            blender_exe = "/project2/yichuanh/blender-4.0.2-linux-x64/blender" # Revise here!

            glb2img_script = "glb2img.py"
            blender_cmd = (
                f"{blender_exe} -b -P {glb2img_script} -- "
                f"--input_dir data/{data_root}/{scene} "
                f"--n_views 100 --res 1200"
            )

            try:
                subprocess.run(blender_cmd, shell=True, check=True)
                print(f"Blender render done: {scene}")
            except subprocess.CalledProcessError:
                print(f"Blender render failed: {scene}")
                os.chdir(original_cwd)
                failed_jobs.append((scene, "Blender Error", ""))
                continue
        else:
            print(f"{scene} already has render data, skipping render step.")

        os.chdir(original_cwd)

        # --- Step 2: Run execute_gumbel.py ---
        for cfg in cfg_list:
            cell_size, palette, color_num = cfg

            print(f"\n[Step 2] Starting Task: scene={scene}, mode={mode}")

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
            # Uncomment to pass mode: if mode: command += f" --mode {mode}"

            print(f"Running:\n{command}\n")

            try:
                subprocess.run(command, shell=True, text=True, check=True)
            except subprocess.CalledProcessError:
                print(f"Failed: scene={scene}, cell_size={cell_size}")
                failed_jobs.append((scene, cell_size, palette))

    print("\nAll tasks completed.")

    if failed_jobs:
        print("\nFailed jobs:")
        for s, c, p in failed_jobs:
            print(f"   Scene: {s}, Info: {c}")
    else:
        print("\nAll succeeded.")

if __name__ == "__main__":
    main()
