import os
from PIL import Image, ImageEnhance
import numpy as np

# Path settings
input_path = './image/output/merged.png'
input_dir = './image/input'   # original renders with transparent background
output_dir = './image/output_bb'
cell_size = 40

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Open image and convert to RGBA mode
img = Image.open(input_path).convert("RGBA")

# Crop settings (2x3 grid)
w_total, h_total = img.size
w = w_total // 3
h = h_total // 2


def enhance_saturation(image, factor=1.5):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def erode_alpha_px(alpha_bool, px):
    """Erode a boolean alpha mask by `px` pixels (4-connected, iterative)."""
    result = alpha_bool.copy()
    for _ in range(px):
        prev = result.copy()
        result[1:, :]  &= prev[:-1, :]
        result[:-1, :] &= prev[1:, :]
        result[:, 1:]  &= prev[:, :-1]
        result[:, :-1] &= prev[:, 1:]
    return result


def cell_mean_saturation(rgb_cell):
    """Return mean HSV saturation (0~1) of a cell's RGB pixels."""
    r, g, b = rgb_cell[:, :, 0] / 255.0, rgb_cell[:, :, 1] / 255.0, rgb_cell[:, :, 2] / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    sat = np.where(cmax > 0, delta / cmax, 0.0)
    return float(np.mean(sat))


def remove_background_cells_eroded(pixel_image, ref_image, cell_size=30, erode_px=0, sat_rescue=0.2):
    """
    Keep cells covered by the eroded input alpha mask.
    Cells that would be removed by erosion are rescued if their pixel art
    saturation exceeds `sat_rescue` (0~1), meaning they carry real color content.
    """
    pixel_array = np.array(pixel_image)
    ref_array = np.array(ref_image.convert("RGBA"))

    if ref_array.shape[:2] != pixel_array.shape[:2]:
        ref_resized = ref_image.convert("RGBA").resize(
            (pixel_array.shape[1], pixel_array.shape[0]), Image.LANCZOS
        )
        ref_array = np.array(ref_resized)

    height, width = pixel_array.shape[:2]
    cell_rows = (height + cell_size - 1) // cell_size
    cell_cols = (width + cell_size - 1) // cell_size

    # Build full alpha mask and eroded alpha mask
    alpha_bool = ref_array[:, :, 3] > 30
    alpha_eroded = erode_alpha_px(alpha_bool, erode_px) if erode_px > 0 else alpha_bool

    mask = np.zeros((height, width), dtype=bool)

    for ci in range(cell_rows):
        for cj in range(cell_cols):
            i, j = ci * cell_size, cj * cell_size

            # Check eroded mask coverage
            cell_alpha_eroded = alpha_eroded[i:i + cell_size, j:j + cell_size]
            if np.mean(cell_alpha_eroded) > 0.3:
                mask[i:i + cell_size, j:j + cell_size] = True
                continue

            # Cell didn't survive erosion — rescue if it's in the original mask
            # AND has high saturation in the pixel art
            cell_alpha_full = alpha_bool[i:i + cell_size, j:j + cell_size]
            if np.mean(cell_alpha_full) > 0.3:
                cell_rgb = pixel_array[i:i + cell_size, j:j + cell_size, :3]
                if cell_mean_saturation(cell_rgb) >= sat_rescue:
                    mask[i:i + cell_size, j:j + cell_size] = True

    new_array = pixel_array.copy()
    new_array[~mask] = [0, 0, 0, 0]
    return Image.fromarray(new_array)


# Crop into six images, then filter each one
for idx in range(6):
    col = idx % 3
    row = idx // 3
    left = col * w
    upper = row * h
    right = left + w
    lower = upper + h

    cropped = img.crop((left, upper, right, lower))

    ref_path = os.path.join(input_dir, f"r_{idx}.png")
    if os.path.exists(ref_path):
        ref_img = Image.open(ref_path)
        cropped = remove_background_cells_eroded(cropped, ref_img, cell_size=cell_size, erode_px=cell_size // 3)
    else:
        print(f"Warning: reference image {ref_path} not found, skipping mask for r_{idx}")

    cropped = enhance_saturation(cropped, 1.25)

    cropped.save(os.path.join(output_dir, f"r_{idx}.png"))
    print(f"Saved r_{idx}.png")

print("Processing and cropping complete!")
