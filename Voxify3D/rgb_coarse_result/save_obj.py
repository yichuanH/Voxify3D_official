import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


def visualize_color_palette(color_palette, save_path, cell_size=100):
    num_colors = color_palette.shape[0]
    image = np.zeros((cell_size, num_colors * cell_size, 3), dtype=np.float32)
    for i, color in enumerate(color_palette):
        image[:, i * cell_size:(i + 1) * cell_size, :] = color
    plt.figure(figsize=(num_colors, 1))
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved color palette to: {save_path}")


def read_voxel_data(filename, palette_file=None):
    folder_path = os.path.dirname(filename)
    data = np.load(filename)
    alpha = data['alpha']
    logit = data['rgb']  # palette logits

    color_palette = None
    if palette_file is not None:
        palette_data = np.load(palette_file)
        color_palette = palette_data['color_palette']
        visualize_color_palette(color_palette, save_path=os.path.join(folder_path, "color_palette.png"))

    return alpha, logit, color_palette


def map_logit_to_rgb(logit, color_palette):
    class_indices = np.argmax(logit, axis=-1)
    rgb = color_palette[class_indices]
    return rgb


def save_voxel_as_obj(filename, alpha, rgb, threshold=0.78, cube_size=1.0): ##cell_size 50以上 0.85佳
    
    with open(filename, 'w') as file:
        file.write("# OBJ file\n")

        cube_vertices = [
            (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)
        ]
        cube_faces = [
            (1, 2, 3, 4), (5, 6, 7, 8), (1, 5, 8, 4),
            (2, 6, 7, 3), (1, 2, 6, 5), (4, 3, 7, 8)
        ]

        vertex_count = 1

        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                for k in range(alpha.shape[2]):
                    if alpha[i, j, k] >= threshold:
                        r, g, b = rgb[i, j, k]
                        for dx, dy, dz in cube_vertices:
                            x = i + dx * cube_size
                            y = j + dy * cube_size
                            z = k + dz * cube_size
                            file.write(f"v {x} {y} {z} {r} {g} {b}\n")
                        for face in cube_faces:
                            v1, v2, v3, v4 = (vertex_count + idx - 1 for idx in face)
                            file.write(f"f {v1} {v2} {v3} {v4}\n")
                        vertex_count += 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert logit voxel grid to OBJ file.')
    parser.add_argument('--input_file', type=str, help='Path to the .npz file')
    parser.add_argument('--output_file', type=str, help='Path to the .obj file')
    parser.add_argument('--palette_file', type=str, help='Path to the color palette .npz file')

    args = parser.parse_args()

    alpha, logit, color_palette = read_voxel_data(args.input_file, args.palette_file)
    
    """
    color_palette = np.array([
    [1.0, 0.0, 0.0],  # red
    [0.0, 1.0, 0.0],  # green
    [0.0, 0.0, 1.0],  # blue
    [1.0, 1.0, 0.0],  # yellow
    [1.0, 0.0, 1.0],  # magenta
    [0.0, 1.0, 1.0],  # cyan
    [1.0, 0.5, 0.0],  # orange
    [0.5, 0.0, 1.0],  # purple
    ])
    """
    
    if color_palette is not None:
        rgb = map_logit_to_rgb(logit, color_palette)
    else:
        rgb = logit  # fallback

    save_voxel_as_obj(args.output_file, alpha, rgb)
