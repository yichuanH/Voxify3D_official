# Voxify3D: Pixel Art Meets Volumetric Rendering

**CVPR 2026**

[**Project Page**](https://yichuanh.github.io/Voxify-3D/) | [**ArXiv**](https://arxiv.org/abs/2512.07834)

<p align="center">
  <img src="teaser.png" width="100%" />
</p>
<p align="center">
  <img src="teaser.gif" width="100%" />
</p>

Official implementation of Voxify3D.

This repository provides the complete pipeline for generating stylized voxel art by combining pixel art guidance with volumetric rendering.

---

## Two Pipelines

This repository contains two entry points depending on your starting material:

| Script | Input | When to use |
|---|---|---|
| `Run_Voxify3D_glb.py` | `.glb` 3D model file | You have a GLB file and want a quick end-to-end result. Blender is required to auto-render orthographic images. |
| `Run_Voxify3d.py` | Pre-rendered orthographic images | You have already prepared 50–100 orthographic-rendered images yourself. |

---

<!-- ## Tested Environment

This project has been tested with the following setup:

| Component | Version |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
| Python | 3.10 |
| CUDA | 11.8 |
| PyTorch | 2.5.0 |
| torchvision | 0.20.0 |
| torchaudio | 2.5.0 |
| mmcv | 2.2.0 |
| Blender | 3.x / 4.x |

> Using other Python / CUDA / PyTorch versions may require additional adjustments, especially for `torch_scatter` and `mmcv` which are tightly coupled to specific CUDA and PyTorch versions.

--- -->

## Environment Setup

We recommend creating a dedicated conda environment named `voxify3d`:

```bash
conda create -n voxify3d python=3.10 -y
conda activate voxify3d
```

> **Before running `pip install -r requirements.txt`**, you must first manually install the following packages that are tightly coupled to your CUDA and PyTorch versions:
> - `torch` / `torchvision` / `torchaudio`
> - `torch_scatter`
> - `mmcv`
> - `torch-efficient-distloss`
>
> These cannot be installed generically — please refer to each project's official installation instructions and match the versions to your environment. Using mismatched versions is the most common cause of setup failures.

Once the above are installed, install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

### Additional requirement for `Run_Voxify3D_glb.py` (GLB pipeline)

`Run_Voxify3D_glb.py` uses **Blender** to automatically render orthographic images from a GLB file. Blender cannot be installed via pip and must be set up separately:

1. Download Blender from https://www.blender.org/download/ (tested with Blender 3.x / 4.x)
2. In `Run_Voxify3D_glb.py`, update `blender_exe` to the absolute path of your Blender executable:

```python
blender_exe = "/path/to/your/blender"
```

---

## Pretrained Models

Please download the following pretrained models and place them in the specified directories.

### PixelArt pretrained models

- **Pixel Art checkpoint** → place under: `PixelArt/`  
  (Download) → [Pixel Art checkpoint](https://drive.google.com/file/d/1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM/view?usp=sharing)

- **AliasNet checkpoint** → place under: `PixelArt/`  
  (Download) → [AliasNet checkpoint](https://drive.google.com/file/d/17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_/view?usp=sharing)

---

### PixelArt auxiliary checkpoints

- **I2PNet checkpoint** → place under: `PixelArt/checkpoints/pixel_model`  
  (Download) → [I2PNet checkpoint](https://drive.google.com/file/d/1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az/view?usp=sharing)

- **P2INet checkpoint** → place under: `PixelArt/checkpoints/pixel_model`  
  (Download) → [P2INet checkpoint](https://drive.google.com/file/d/1z9SmQRPoIuBT_18mzclEd1adnFn2t78T/view?usp=sharing)

---

## Pipeline 1: GLB to Voxel Art (`Run_Voxify3D_glb.py`)

This pipeline takes a `.glb` file as input and handles orthographic rendering automatically using Blender.

Before running, edit `scene_configs` at the top of `Run_Voxify3D_glb.py` to specify which scenes to process and their parameters (`cell_size`, `palette_mode`, `color_num`).

Place your `.glb` file under `Voxify3D/data/{data_root}/{scene}/` and name it `{scene}.glb`. For example:

```text
Voxify3D/data/GLB/fallguy/fallguy.glb
```

Then run:

```bash
python Run_Voxify3D_glb.py --device 0 --data_root GLB
```

- `--device`: GPU id(s), e.g. `0` or `0,1`
- `--data_root`: root directory for the data (default: `GLB`)

#### Checking render quality

After the first run, check whether the object appears too large or too small in the rendered images. If it does, adjust `WORLD_SIZE` in `Voxify3D/glb2img.py`:

```python
WORLD_SIZE = 2.0  # Increase to zoom out, decrease to zoom in
```

Then delete the incorrect render folders (e.g. `ortho/`, `6views/`) — keep only the `.glb` file in the scene directory — and re-run the pipeline.

#### Output

Results are saved under `Voxify3D/voxel_result/`.

---

## Pipeline 2: Pre-rendered Images to Voxel Art (`Run_Voxify3d.py`)

Use this pipeline if you have already prepared your own **50–100 orthographic-rendered images** with known camera parameters.

Edit the configuration section in `Run_Voxify3d.py`:

```python
data_root = "Rodin"

scene_configs = {
    "Dragon": 25,
    # "redpanda": 50,
}

color_nums = [6]
palette_modes = ["kmeans"]  # "maxmin", "mediancut", "sa"
```

Here, `data_root` specifies the dataset root directory, `scene_configs` maps scene names to voxel cell sizes, `color_nums` sets how many palette colors to use, and `palette_modes` selects the palette generation method(s).

Run the pipeline with:

```bash
python Run_Voxify3d.py --gpu 0
```

The `--gpu` argument specifies the GPU id via `CUDA_VISIBLE_DEVICES`.

---

## Notes

This codebase has been tested on Linux systems with NVIDIA GPUs. Please ensure that all pretrained models and dataset files are placed in the correct directories before running the pipeline.

---

## Acknowledgement

Our pixel art pipeline and models are based on  
*Make Your Own Sprites: Aliasing-Aware and Cell-Controllable Pixelization* (ACM TOG).

The core voxel rendering pipeline is built upon  
*Direct Voxel Grid Optimization*.

The dataset used in this project is derived from  
*Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion*.

We thank the authors for making their work publicly available.

---

## Citation

If you find our work useful, please cite this paper and give us a ⭐️.

```BibTex
@inproceedings{huang2026voxify3d,
  author    = {Huang, Yi-Chuan and Chan, Jiewen and Chien, Hao-Jen and Liu, Yu-Lun},
  title     = {Voxify3D: Pixel Art Meets Volumetric Rendering},
  booktitle = {CVPR},
  year      = {2026}
}
```
