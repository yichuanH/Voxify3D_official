import os
from PIL import Image
import numpy as np

input_dir = './image/input'
output_dir = './image/input_bb'
output_path = os.path.join(output_dir, 'merged.png')

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read six images (sorted by filename)
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])[:6]

images = []

# Apply white background
for filename in image_files:
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path).convert("RGBA")

    # Create white background
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white_bg.paste(img, (0, 0), mask=img)  # Paste original image onto white background (preserve transparency mask)

    images.append(white_bg.convert("RGB"))  # Convert to RGB

# Ensure there are exactly six images
if len(images) != 6:
    raise ValueError("Please ensure the input folder contains exactly 6 PNG images!")

# Assume all images are the same size
w, h = images[0].size

# Create a 2x3 grid image (3 images per row, 2 rows)
grid_img = Image.new('RGB', (3 * w, 2 * h), (255, 255, 255))

# Paste images into grid
for idx, img in enumerate(images):
    x = (idx % 3) * w
    y = (idx // 3) * h
    grid_img.paste(img, (x, y))

# Save merged image
grid_img.save(output_path)
print(f"Saved merged image: {output_path}")
