# Voxify3D: Pixel Art Meets Volumetric Rendering

[**Project Page**](https://yichuanh.github.io/Voxify-3D/) ｜ [**ArXiv**](https://arxiv.org/abs/2512.07834)

<p align="center">
  <img src="teaser.png" width="100%" />
</p>
<p align="center">
  <img src="teaser.gif" width="100%" />
</p>

Official implementation of Voxify3D.

This repository provides the complete pipeline for generating stylized voxel art by combining pixel art guidance with volumetric rendering.

---

## 🛠️ TODO
- ✅ Release a version that requires users to prepare the dataset format themselves (orthographic, 6 views)
- ✅ Provide an example dataset: `Rodin/Dragon`
- ✅ Provide a simplified GLB-to-voxel-art pipeline (`Run_Voxify3D_glb.py`)
- ⬜ Provide more example datasets
> 🔔 **Note:** A fully integrated version with a one-click end-to-end pipeline is planned to be released in **2026 February**.

---

## Two Pipelines

This repository contains two entry points depending on your starting material:

| Script | Input | When to use |
|---|---|---|
| `Run_Voxify3D_glb.py` | `.glb` 3D model file | You have a GLB file and want a quick end-to-end result. Blender is required to auto-render orthographic images. |
| `Run_Voxify3d.py` | Pre-rendered orthographic images | You have already prepared 50–100 orthographic-rendered images yourself. |

---

## Environment Setup

We recommend creating a dedicated conda environment named `voxify3d`.

Create and activate the environment:

```bash
conda create -n voxify3d python=3.9 -y
conda activate voxify3d
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch manually according to your CUDA version. Please refer to the official PyTorch installation guide:  
https://pytorch.org/get-started/locally/

Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Additional requirement for `Run_Voxify3D_glb.py` (GLB pipeline)

`Run_Voxify3D_glb.py` uses **Blender** to automatically render orthographic images from a GLB file. Blender cannot be installed via pip and must be set up separately:

1. Download Blender from https://www.blender.org/download/ (tested with Blender 3.x / 4.x)
2. Make sure the `blender` executable is accessible from your terminal (add it to `PATH`), or set the path directly in the script.

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

Edit the scene configuration at the top of `Run_Voxify3D_glb.py`:

```python
scene_configs = {
    "fallguy": [[30, "kmeans_rare", 8]],
    # "redpanda": [[50, "kmeans", 6]],
}
```

Each entry maps a scene name to a list of `[cell_size, palette_mode, color_num]` configs.

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

## Preparing Your Own Dataset (for `Run_Voxify3d.py`)

### Orthographic multi-view images

Prepare **50–100 orthographic-rendered images** of the mesh with known camera parameters and place them under:

```text
Voxify3D/data/{ROOT}/{scene}/ortho/
```

The directory must contain:

```text
ortho/
├── train/
├── test/
├── val/
├── transforms_train.json
├── transforms_test.json
└── transforms_val.json
```

- Images should be rendered using **orthographic projection**
- `train`, `test`, and `val` contain RGB images
- Each `transforms_*.json` stores the corresponding camera parameters

---

### Canonical 6-view orthographic projections

Additionally, prepare **six canonical orthographic views** (front, back, left, right, top, bottom) and place them under:

```text
Voxify3D/data/{ROOT}/{scene}/6views/
```

The directory must contain:

```text
6views/
├── train/
├── test/
├── val/
├── transforms_train.json
├── transforms_test.json
└── transforms_val.json
```

- The `train`, `test`, and `val` folders may contain identical images
- The transforms files should include camera parameters for the six canonical views

---

### Reference example

A complete data format example is provided at:

```text
Voxify3D/data/Rodin/Dragon
```

We strongly recommend following this example when preparing custom datasets.

---

## Notes

This codebase has been tested on Linux systems with NVIDIA GPUs. Different CUDA or PyTorch versions may require minor adjustments. Please ensure that all pretrained models and dataset files are placed in the correct directories before running the pipeline.

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
@article{voxify3d,
    author    = {Huang, Yi-Chuan and Chan, Jiewen and Chien, Hao-Jen and Liu, Yu-Lun},
    title     = {Voxify3D: Pixel Art Meets Volumetric Rendering},
    journal   = {arXiv preprint arXiv:2512.07834},
    year      = {2025}
}
```
