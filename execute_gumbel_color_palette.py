#!/usr/bin/env python3
import os
import argparse
import subprocess
import time


def set_gpu_env(gpu: str, launch_blocking: bool = False) -> None:
    # Set ASAP
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def run_command(command: str, gpu: str) -> None:
    """
    Run a shell command pinned to a specific GPU via CUDA_VISIBLE_DEVICES.
    This mimics the 'full_command = CUDA_VISIBLE_DEVICES=1 ...' style.
    """
    full_command = f"CUDA_VISIBLE_DEVICES={gpu} {command}"
    print(f"[RUN] {full_command}")
    result = subprocess.run(full_command, shell=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed (returncode={result.returncode}): {full_command}")
        raise SystemExit(result.returncode)


def format_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.2f}s"


def main(scene: str, cell_size: int, color_num: int, palette_mode: str,
         mode: str | None, data_root: str, gpu: str) -> None:

    total_start = time.time()
    durations: dict[str, float] = {}

    vox_num = int((1200 / cell_size) ** 3)
    print(f"[INFO] scene={scene}, cell_size={cell_size}, vox_num={vox_num}, "
          f"color_num={color_num}, palette_mode={palette_mode}, gpu={gpu}, mode={mode}")

    # ===================== PixelArt stage (only when mode is None) =====================
    if mode is None:
        stage_start = time.time()

        run_command("rm -rf PixelArt/image/input/*", gpu)

        # NOTE: Keep your original Voxify3D paths/behavior
        run_command(
            f"cp -r Voxify3D/data/{data_root}/{scene}/6views "
            f"Voxify3D/data/{data_root}/{scene}/6views_n",
            gpu,
        )
        run_command(
            f"cp Voxify3D/data/{data_root}/{scene}/6views_n/train/* PixelArt/image/input/",
            gpu,
        )

        os.chdir("PixelArt")
        run_command("python pre_process.py", gpu)
        run_command(
            f"python test_pro.py --input ./image/input_bb --cell_size {cell_size} "
            f"--model_name pixel_model --output ./image/output/",
            gpu,
        )
        run_command(
            f"sed -i 's/^cell_size = .*/cell_size = {cell_size}/' post_process.py",
            gpu,
        )
        run_command(f"python post_process.py --cell_size {cell_size}", gpu)
        os.chdir("..")

        run_command(
            f"cp PixelArt/image/output_bb/* Voxify3D/data/{data_root}/{scene}/6views/train/",
            gpu,
        )
        run_command(f"mkdir -p Voxify3D/data/{data_root}/{scene}/6views/test/", gpu)
        run_command(f"mkdir -p Voxify3D/data/{data_root}/{scene}/6views/val/", gpu)
        run_command(
            f"cp PixelArt/image/output_bb/* Voxify3D/data/{data_root}/{scene}/6views/test/",
            gpu,
        )
        run_command(
            f"cp PixelArt/image/output_bb/* Voxify3D/data/{data_root}/{scene}/6views/val/",
            gpu,
        )

        run_command(
            f"cp Voxify3D/data/our_data/transforms_train.json "
            f"Voxify3D/data/{data_root}/{scene}/6views/",
            gpu,
        )
        run_command(
            f"cp Voxify3D/data/our_data/transforms_train.json "
            f"Voxify3D/data/{data_root}/{scene}/6views/transforms_test.json",
            gpu,
        )
        run_command(
            f"cp Voxify3D/data/our_data/transforms_train.json "
            f"Voxify3D/data/{data_root}/{scene}/6views/transforms_val.json",
            gpu,
        )

        durations["PixelArt"] = time.time() - stage_start
        print("[INFO] ------------------ Voxify3D Start -----------------")

    # ===================== Ortho training =====================
    if mode in (None, "train_ortho", "train_o+6"):
        stage_start = time.time()

        log_dir = f"Voxify3D/logs/{data_root}/{scene}/"
        run_command(f"rm -rf {log_dir}/*", gpu)
        run_command(f"mkdir -p {log_dir}", gpu)

        os.chdir("Voxify3D")

        run_command(f"mkdir -p configs/{data_root}/{scene}", gpu)
        run_command(
            f"cp configs/z_default/z_ortho.py configs/{data_root}/{scene}/{scene}_ortho.py",
            gpu,
        )
        run_command(
            f"cp configs/z_default/z_6views.py configs/{data_root}/{scene}/{scene}_6views.py",
            gpu,
        )

        run_command(
            f"sed -i 's/car/{scene}/g' configs/{data_root}/{scene}/{scene}_ortho.py",
            gpu,
        )
        run_command(
            f"sed -i 's/car/{scene}/g' configs/{data_root}/{scene}/{scene}_6views.py",
            gpu,
        )
        run_command(
            f"sed -i 's/our_data/{data_root}/g' configs/{data_root}/{scene}/{scene}_ortho.py",
            gpu,
        )
        run_command(
            f"sed -i 's/our_data/{data_root}/g' configs/{data_root}/{scene}/{scene}_6views.py",
            gpu,
        )

        run_command(f"cp configs/z_default/default_ortho.py configs/{data_root}/default_ortho.py", gpu)
        run_command(f"cp configs/z_default/default_6views.py configs/{data_root}/default_6views.py", gpu)

        run_command(f"cp data/z_default/transforms_test.json  data/{data_root}/transforms_test.json", gpu)
        run_command(f"cp data/z_default/transforms_train.json data/{data_root}/transforms_train.json", gpu)
        run_command(f"cp data/z_default/transforms_val.json   data/{data_root}/transforms_val.json", gpu)

        run_command(
            f"sed -i 's/num_voxels_ini = [0-9]*/num_voxels_ini = {vox_num}/' "
            f"configs/{data_root}/default_ortho.py",
            gpu,
        )
        run_command(
            f"sed -i 's/num_voxels_ini = [0-9]*/num_voxels_ini = {vox_num}/' "
            f"configs/{data_root}/default_6views.py",
            gpu,
        )

        run_command(f"rm -rf data/{data_root}/{scene}/ortho/test/*", gpu)
        run_command(f"mkdir -p data/{data_root}/{scene}/ortho/test/", gpu)
        run_command(
            f"cp data/{data_root}/{scene}/6views_n/train/* data/{data_root}/{scene}/ortho/test/",
            gpu,
        )
        run_command(
            f"cp data/{data_root}/transforms_test.json data/{data_root}/{scene}/ortho/transforms_test.json",
            gpu,
        )

        run_command(
            f"python3 run_ortho.py --config configs/{data_root}/{scene}/{scene}_ortho.py --render_test",
            gpu,
        )
        run_command(
            f'python3 run_ortho.py --config configs/{data_root}/{scene}/{scene}_ortho.py '
            f'--export_coarse_only "rgb_coarse_result/{data_root}/{scene}/{scene}_ortho/{scene}_ortho"',
            gpu,
        )

        os.chdir("rgb_coarse_result/")
        run_command(
            f'python3 save_obj_ori.py --input_file "{data_root}/{scene}/{scene}_ortho/{scene}_ortho.npz" '
            f'--output_file "{data_root}/{scene}/{scene}_ortho/{scene}_ortho.obj" '
            f'--uncertainty_file "{data_root}/{scene}/{scene}_ortho/{scene}_uncertainty.obj"',
            gpu,
        )
        os.chdir("..")

        run_command(
            f"cp logs/{data_root}/{scene}/dvgo_{scene}_ortho/render_test_fine_last/test_depths.npz "
            f"data/{data_root}/{scene}/6views/test_depths.npz",
            gpu,
        )

        os.chdir("..")  # leave Voxify3D

        durations["Ortho"] = time.time() - stage_start

    # ===================== 6views training (color palette) =====================
    if mode in (None, "train_6views", "train_o+6", "train_o+6+od", "train_6+od"):
        stage_start = time.time()

        os.chdir("Voxify3D")

        run_command(
            f'python3 run_6views_color_palette.py --config configs/{data_root}/{scene}/{scene}_6views.py '
            f'--ft_path "logs/{data_root}/{scene}/dvgo_{scene}_ortho/coarse_last.tar" '
            f'--render_test --color_num {color_num} --palette_mode {palette_mode}',
            gpu,
        )

        run_command(
            f"mkdir -p rgb_coarse_result/{data_root}/{scene}/{scene}_6views && "
            f"cp logs/{data_root}/{scene}/dvgo_{scene}6_views/color_palette.npz "
            f"rgb_coarse_result/{data_root}/{scene}/{scene}_6views/color_palette.npz",
            gpu,
        )

        run_command(
            f'python3 run_6views_color_palette.py --config configs/{data_root}/{scene}/{scene}_6views.py '
            f'--export_coarse_only "rgb_coarse_result/{data_root}/{scene}/{scene}_6views/{scene}_6views" '
            f'--palette_mode {palette_mode}',
            gpu,
        )

        os.chdir("rgb_coarse_result/")
        run_command(
            f'python3 save_obj.py --input_file "{data_root}/{scene}/{scene}_6views/{scene}_6views.npz" '
            f'--output_file "{data_root}/{scene}/{scene}_6views/{scene}_6views.obj" '
            f'--palette_file "{data_root}/{scene}/{scene}_6views/color_palette.npz"',
            gpu,
        )
        os.chdir("..")
        os.chdir("..")  # leave Voxify3D

        out_dir = f"Voxify3D/voxel_result/{data_root}/{scene}/{palette_mode}/{color_num}/{cell_size}"
        run_command(f"mkdir -p {out_dir}", gpu)

        run_command(
            f"cp Voxify3D/rgb_coarse_result/{data_root}/{scene}/{scene}_6views/{scene}_6views.obj "
            f"{out_dir}/{scene}_{palette_mode}_{color_num}_{cell_size}_gumbel.obj",
            gpu,
        )
        run_command(
            f"cp Voxify3D/rgb_coarse_result/{data_root}/{scene}/{scene}_6views/color_palette.png "
            f"{out_dir}/color_palette.png",
            gpu,
        )

        durations["6Views"] = time.time() - stage_start
        print("~~~ project over ❤️ ~~~")

    # ===================== Summary =====================
    print("\nExecution Summary:")
    total_time = time.time() - total_start
    for stage, dur in durations.items():
        print(f"  - {stage}: {format_time(dur)}")
    print(f"Total time: {format_time(total_time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxify3D pipeline runner")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--cell_size", type=int, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["train_ortho", "train_6views", "train_o+6", "train_6order",
                 "train_o+6+od", "train_6+od", "save"],
    )
    parser.add_argument("--data_root", type=str, required=True)

    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES id")
    parser.add_argument("--launch_blocking", action="store_true", help="Set CUDA_LAUNCH_BLOCKING=1")

    parser.add_argument("--color_num", type=int, required=True)
    parser.add_argument("--palette_mode", type=str, required=True,
                        choices=["kmeans", "kmeans_rare", "maxmin", "mediancut", "sa"])

    args = parser.parse_args()

    set_gpu_env(args.gpu, launch_blocking=args.launch_blocking)

    main(
        scene=args.scene,
        cell_size=args.cell_size,
        color_num=args.color_num,
        palette_mode=args.palette_mode,
        mode=args.mode,
        data_root=args.data_root,
        gpu=args.gpu,
    )
