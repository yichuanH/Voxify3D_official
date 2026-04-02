import os
from PIL import Image, ImageEnhance
import numpy as np

# Path settings
input_path = './image/output/merged.png'
output_dir = './image/output_bb'

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Open image and convert to RGBA mode
img = Image.open(input_path).convert("RGBA")
img_array = np.array(img)


def enhance_saturation(image, factor=1.5):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)



def enhance_contrast(image, factor=1.5):
    """
    Enhance overall contrast.
    factor > 1 increases contrast, factor < 1 decreases contrast.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


# Create mask: white (too bright) pixels -> set to transparent
bright_mask = np.all(img_array[:, :, :3] > [220, 220, 220], axis=-1)

# Set matching pixels to transparent
img_array[bright_mask] = [0, 0, 0, 0]

# Save intermediate processed image (optional)
processed_img = Image.fromarray(img_array)

# Crop settings (2x3 grid)
w_total, h_total = processed_img.size
w = w_total // 3
h = h_total // 2

processed_img = enhance_saturation(processed_img, 1.25)
#processed_img = enhance_contrast(processed_img, 1.25)

# Crop into six images and name them
for idx in range(6):
    col = idx % 3
    row = idx // 3
    left = col * w
    upper = row * h
    right = left + w
    lower = upper + h

    cropped = processed_img.crop((left, upper, right, lower))
    cropped.save(os.path.join(output_dir, f"r_{idx}.png"))
    print(f"Saved r_{idx}.png")

print("Processing and cropping complete!")
