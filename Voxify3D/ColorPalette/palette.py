import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import random
import colorsys
from scipy.spatial.distance import cdist


def is_unsaturated(color_rgb, sat_threshold=0.2):
    """Check if an RGB color has low saturation."""
    r, g, b = color_rgb
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return s < sat_threshold

def boost_saturation(rgb, factor=2):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(s * factor, 1.0)
    r_new, g_new, b_new = colorsys.hsv_to_rgb(h, s, v)
    return np.array([r_new, g_new, b_new])




def save_color_palette(palette, save_path="ColorPalette/color/palette.png", square_size=64):
    """
    Draw the palette as a single-row color swatch image and save it.

    Args:
        palette (np.ndarray): shape = (C, 3), RGB colors in [0, 1]
        save_path (str): save location
        square_size (int): side length (pixels) of each color square
    """
    C = palette.shape[0]
    palette_uint8 = (palette * 255).astype(np.uint8)

    canvas = np.zeros((square_size, C * square_size, 3), dtype=np.uint8)
    for i in range(C):
        canvas[:, i*square_size:(i+1)*square_size, :] = palette_uint8[i]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = Image.fromarray(canvas)
    img.save(save_path)
    print(f"Palette saved to: {save_path}")

def median_cut(pixels, C):
    """Run median cut on pixels and return C representative colors."""
    def split_box(box):
        pixels = box['pixels']
        ranges = np.ptp(pixels, axis=0)  # range of each channel
        split_dim = np.argmax(ranges)  # find channel with largest range

        # sort by that channel and split in half
        sorted_pixels = pixels[pixels[:, split_dim].argsort()]
        median_index = len(sorted_pixels) // 2
        return [
            {'pixels': sorted_pixels[:median_index]},
            {'pixels': sorted_pixels[median_index:]},
        ]

    # start with one box
    boxes = [{'pixels': pixels}]
    while len(boxes) < C:
        # split the box with the most pixels
        boxes.sort(key=lambda b: len(b['pixels']), reverse=True)
        box_to_split = boxes.pop(0)
        new_boxes = split_box(box_to_split)
        boxes.extend(new_boxes)

    # take average color of each box
    palette = np.array([np.mean(b['pixels'], axis=0) for b in boxes])
    return palette

def choose_colors_median_cut(images, C):
    all_pixels = []

    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        assert img.shape[-1] == 3, "Image must be in RGB format"
        pixels = img.reshape(-1, 3)

        # filter out white pixels
        non_white_mask = ~(np.all(np.isclose(pixels, 1.0, atol=1e-4), axis=1))
        valid_pixels = pixels[non_white_mask]
        all_pixels.append(valid_pixels)

    all_pixels = np.concatenate(all_pixels, axis=0)

    if len(all_pixels) == 0:
        raise ValueError("All pixels were filtered out. Please verify the input images.")

    # run median cut
    palette = median_cut(all_pixels, C)

    # save palette
    save_color_palette(palette, save_path="ColorPalette/color/palette_mediancut.png")

    return palette


def maxmin_picking(pixels, C):
    # deduplicate before picking
    pixels = np.unique(pixels, axis=0)

    # randomly pick the first color
    idx = np.random.randint(len(pixels))
    selected = [pixels[idx]]

    for _ in range(C - 1):
        # compute min distance from each pixel to all selected colors
        dists = np.array([
            np.min([np.linalg.norm(p - s) for s in selected])
            for p in pixels
        ])

        # pick the farthest pixel
        farthest_idx = np.argmax(dists)
        selected.append(pixels[farthest_idx])

    return np.stack(selected)


from scipy.spatial.distance import cdist

def choose_colors_kmeans_with_rare(images, C, rare_color_count=None, alpha=0.8, beta=1.0):

    if C < 2:
        raise ValueError("C must be >= 2")

    # auto-compute rare color count: default int(C/4), at least 1, less than C
    if rare_color_count is None:
        rare_color_count = max(1, min(C // 4, C - 1))
    assert C > rare_color_count, "C must be greater than rare_color_count"

    all_pixels = []

    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0  # convert to [0,1]

        assert img.shape[-1] == 3, "Image must be in RGB format"
        pixels = img.reshape(-1, 3)

        non_white_mask = ~(np.all(np.isclose(pixels, 1.0, atol=1e-4), axis=1))
        valid_pixels = pixels[non_white_mask]
        all_pixels.append(valid_pixels)

    all_pixels = np.concatenate(all_pixels, axis=0)  # shape (N, 3)

    if len(all_pixels) == 0:
        raise ValueError("All pixels were filtered out.")

    # -- Step 1: K-means for dominant colors --
    kmeans_main = KMeans(n_clusters=C - rare_color_count, random_state=0, n_init='auto')
    kmeans_main.fit(all_pixels)
    palette = list(kmeans_main.cluster_centers_)

    # -- Step 2: compute rarity score for each pixel --
    bins = 32
    indices = np.floor(all_pixels * bins).clip(0, bins - 1).astype(int)
    bin_keys = indices[:, 0] * bins**2 + indices[:, 1] * bins + indices[:, 2]
    bin_counts = np.bincount(bin_keys, minlength=bins**3)
    pixel_bin_counts = bin_counts[bin_keys]

    rare_score = 1.0 / (pixel_bin_counts.astype(np.float32) + 1e-6)
    rare_score = (rare_score - rare_score.min()) / (rare_score.max() - rare_score.min() + 1e-6)

    selected_mask = np.zeros(len(all_pixels), dtype=bool)  # prevent re-selecting same pixel

    # -- Step 3: pick one rare color at a time, dynamically updating palette --
    for _ in range(rare_color_count):
        palette_np = np.stack(palette, axis=0)  # shape (P, 3)

        dists = cdist(all_pixels, palette_np)
        color_score = np.min(dists, axis=1)
        color_score = (color_score - color_score.min()) / (color_score.max() - color_score.min() + 1e-6)

        # final score = alpha * rare_score + beta * distance
        final_score = alpha * rare_score + beta * color_score

        # exclude already-selected pixels
        final_score[selected_mask] = -np.inf

        best_idx = np.argmax(final_score)
        best_color = all_pixels[best_idx]
        palette.append(best_color)
        selected_mask[best_idx] = True

    # -- Output --
    final_palette = np.stack(palette, axis=0)
    save_color_palette(final_palette, save_path="ColorPalette/color/palette.png")

    return final_palette



def choose_colors_maxmin(images, C):
    all_pixels = []

    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        assert img.shape[-1] == 3, "Image must be in RGB format"
        pixels = img.reshape(-1, 3)

        # filter out white pixels
        non_white_mask = ~(np.all(np.isclose(pixels, 1.0, atol=1e-4), axis=1))
        valid_pixels = pixels[non_white_mask]
        all_pixels.append(valid_pixels)
        pixels = np.unique(pixels, axis=0)  # remove duplicate colors


    all_pixels = np.concatenate(all_pixels, axis=0)

    if len(all_pixels) == 0:
        raise ValueError("All pixels were filtered out. Please verify the input images.")

    # run MaxMin Picking
    palette = maxmin_picking(all_pixels, C)

    # save palette
    save_color_palette(palette, save_path="ColorPalette/color/palette_maxmin.png")

    return palette


def choose_colors_kmeans(images, C):
    all_pixels = []

    #breakpoint()

    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0  # convert to [0,1]

        assert img.shape[-1] == 3, "Image must be in RGB format"
        pixels = img.reshape(-1, 3)  # (H*W, 3)

        # ignore pure white pixels (with small tolerance)
        non_white_mask = ~(np.all(np.isclose(pixels, 1.0, atol=1e-4), axis=1))
        valid_pixels = pixels[non_white_mask]

        all_pixels.append(valid_pixels)

    all_pixels = np.concatenate(all_pixels, axis=0)  # (N_total, 3)

    if len(all_pixels) == 0:
        raise ValueError("All pixels were filtered out. Please verify the input images.")
    #breakpoint()

    # run K-means
    kmeans = KMeans(n_clusters=C, random_state=0, n_init='auto')
    kmeans.fit(all_pixels)

    palette = kmeans.cluster_centers_
    save_color_palette(palette, save_path="ColorPalette/color/palette.png")

    #breakpoint()

    return palette  # shape (C, 3)



def simulated_annealing(pixels, C, max_iter=500, T_start=1.0, T_end=1e-3, alpha=0.95):
    """Select C colors from pixels using Simulated Annealing."""
    N = len(pixels)
    if N <= C:
        return pixels[:C]

    def cost(palette):
        # sum of each pixel's distance to its nearest palette color
        dists = np.min(np.linalg.norm(pixels[:, None, :] - palette[None, :, :], axis=2), axis=1)
        return np.mean(dists)

    # initial palette
    current_palette = pixels[np.random.choice(N, C, replace=False)]
    current_cost = cost(current_palette)
    best_palette = current_palette.copy()
    best_cost = current_cost

    T = T_start
    for it in range(max_iter):
        # randomly replace one point in the palette
        new_palette = current_palette.copy()
        idx = np.random.randint(C)
        new_palette[idx] = pixels[np.random.randint(N)]

        new_cost = cost(new_palette)
        delta = new_cost - current_cost

        if delta < 0 or np.exp(-delta / T) > random.random():
            current_palette = new_palette
            current_cost = new_cost
            if current_cost < best_cost:
                best_palette = current_palette.copy()
                best_cost = current_cost

        T = max(T * alpha, T_end)  # cool down

    return best_palette


def choose_colors_sa(images, C):
    all_pixels = []

    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        assert img.shape[-1] == 3, "Image must be in RGB format"
        pixels = img.reshape(-1, 3)

        # filter out white pixels
        non_white_mask = ~(np.all(np.isclose(pixels, 1.0, atol=1e-4), axis=1))
        valid_pixels = pixels[non_white_mask]
        all_pixels.append(valid_pixels)

    all_pixels = np.concatenate(all_pixels, axis=0)
    all_pixels = np.unique(all_pixels, axis=0)  # reduce duplicates to shrink search space

    if len(all_pixels) == 0:
        raise ValueError("All pixels were filtered out. Please verify the input images.")

    # run Simulated Annealing
    palette = simulated_annealing(all_pixels, C)

    # save palette
    save_color_palette(palette, save_path="ColorPalette/color/palette_sa.png")

    return palette


def read_hex(hex_path):

    # read all hex colors from file (one per line)
    with open(hex_path, 'r') as f:
        lines = f.readlines()

    # strip blank lines and comments, keep valid hex strings
    hex_colors = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

    # convert hex to RGB array in [0, 1] range
    rgb_array = np.array([
        [int(color[i:i+2], 16) for i in (0, 2, 4)]  # extract R, G, B
        for color in hex_colors
    ]) / 255.0

    return rgb_array
