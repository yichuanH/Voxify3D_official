import os
from PIL import Image
import numpy as np

# Input and output folder paths
input_dir = './image/output'
output_dir = './image/output_bb'

# Set cell size
cell_size = 40

# Create output folder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read all image files
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):  # Only process PNG images
        # Get full path of the image
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)  # Set output path

        # Open image and convert to RGBA mode
        img = Image.open(input_path).convert("RGBA")

        # Convert image to numpy array
        img_array = np.array(img)

        # Image dimensions
        height, width = img_array.shape[:2]

        # Create a mask matrix the same size as the image
        mask = np.ones((height, width), dtype=bool)  # Initialize to keep all

        # Iterate over all cell center points
        for i in range(0, height, cell_size):
            for j in range(0, width, cell_size):
                # Check if current cell is pure white
                cell = img_array[i:i + cell_size, j:j + cell_size, :3]
                if np.all(cell >= [220, 220, 220]):  # Condition for pure white cell
                    # Check surrounding eight neighboring cells
                    white_neighbor_count = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue  # Skip current cell
                            ni, nj = i + di * cell_size, j + dj * cell_size
                            if 0 <= ni < height and 0 <= nj < width:
                                neighbor = img_array[ni:ni + cell_size, nj:nj + cell_size, :3]
                                if np.all(neighbor[:, :, :3] >= [230, 230, 230]) or not mask[ni:ni + cell_size, nj:nj + cell_size].any():
                                    white_neighbor_count += 1
                                    if cell_size<=40 :
                                        if white_neighbor_count >= 4:  # At least four pure white neighbors
                                            mask[i:i + cell_size, j:j + cell_size] = False
                                            break #### remove this
                                    else:
                                        if white_neighbor_count >= 1:
                                            mask[i:i + cell_size, j:j + cell_size] = False
                                            break
                        if not mask[i:i + cell_size, j:j + cell_size].any():
                            break

                if np.all(cell >= [240, 240, 240]):
                    mask[i:i + cell_size, j:j + cell_size] = False
                    continue



        # Iterate over all cell center points
        for i in range(0, height, cell_size):
            for j in range(0, width, cell_size):
                # Check if current cell is highlighted (above 170)
                cell = img_array[i:i + cell_size, j:j + cell_size, :3]
                if np.all(cell >= [200, 200, 200]):  # All pixels in cell >= 200
                    mask[i:i + cell_size, j:j + cell_size] = False



        # Set transparent parts to [0, 0, 0, 0]
        new_img_array = img_array.copy()
        new_img_array[~mask] = [0, 0, 0, 0]

        # Convert modified array back to Image object
        img_result = Image.fromarray(new_img_array)

        # Save modified image to output_bb folder
        img_result.save(output_path)

        print(f"Processed {filename}")

print("All images processed!")
