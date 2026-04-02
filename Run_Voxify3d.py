#!/usr/bin/env python3
import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for Voxify3D (execute_gumbel_color_palette.py)."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU id for CUDA_VISIBLE_DEVICES (e.g., 0,1,2).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Optional mode passed to execute_gumbel_color_palette.py "
             "(e.g., train_ortho, train_6views, train_o+6, ...).",
    )
    args = parser.parse_args()

    # ============================================================
    # Global configuration (edit here)
    # ============================================================
    data_root = "Check"

    scene_configs = {
        "corgi": 30,
        # "bob": 40,
        # "redpanda": 50,
        # "alpaca": 40,
        # "labubu": 40,
    }

    color_nums = [6]
    palette_modes = ["kmeans_rare"] # "maxmin", "mediancut", "sa"
    # ============================================================

    data_path = Path(f"Voxify3D/data/{data_root}/")
    if not data_path.exists():
        print(f"[ERROR] Data folder not found: {data_path}")
        sys.exit(1)

    failed = []

    for scene, cell_size in scene_configs.items():
        for color_num in color_nums:
            for palette_mode in palette_modes:
                print(
                    f"\n=== Running scene={scene}, cell_size={cell_size}, "
                    f"color_num={color_num}, palette_mode={palette_mode}, "
                    f"data_root={data_root}, gpu={args.gpu} ===\n"
                )

                cmd = [
                    sys.executable,
                    "execute_gumbel_color_palette.py",
                    "--scene", scene,
                    "--cell_size", str(cell_size),
                    "--data_root", data_root,
                    "--gpu", str(args.gpu),
                    "--color_num", str(color_num),
                    "--palette_mode", str(palette_mode),
                ]

                if args.mode is not None:
                    cmd += ["--mode", args.mode]

                print("[CMD]", " ".join(cmd))

                try:
                    subprocess.run(cmd, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(
                        f"[ERROR] Failed: scene={scene}, cell_size={cell_size}, "
                        f"color_num={color_num}, palette_mode={palette_mode} "
                        f"(returncode={e.returncode})"
                    )
                    failed.append((scene, cell_size, color_num, palette_mode))
                    sys.exit(1)  # change to `continue` if you want best-effort

    print("\nAll runs finished.")

    if failed:
        print("\nFailed runs:")
        for scene, cell_size, color_num, palette_mode in failed:
            print(
                f"  - scene={scene}, cell_size={cell_size}, "
                f"color_num={color_num}, palette_mode={palette_mode}"
            )
        sys.exit(1)

    print("\nAll runs succeeded.")


if __name__ == "__main__":
    main()
