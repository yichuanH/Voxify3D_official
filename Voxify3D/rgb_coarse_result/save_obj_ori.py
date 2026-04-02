import numpy as np
import argparse
from scipy.stats import rankdata
import matplotlib.pyplot as plt

######################### --------------- This version saves uncertainty! ----------- ##########


def read_voxel_data(filename):
    """Load voxel data and retrieve alpha and rgb values."""
    data = np.load(filename)
    alpha = data['alpha']
    rgb = data['rgb']
    uncertainty = data['uncertainty']

    #breakpoint()
    return alpha, rgb, uncertainty

def has_adjacent_voxel(alpha, i, j, k, threshold):
    """Check if voxel (i, j, k) has at least one neighbor above threshold."""
    shape = alpha.shape
    neighbors = [
        (i - 1, j, k), (i + 1, j, k),  # left/right
        (i, j - 1, k), (i, j + 1, k),  # up/down
        (i, j, k - 1), (i, j, k + 1)   # front/back
    ]

    for ni, nj, nk in neighbors:
        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:  # ensure valid index
            if alpha[ni, nj, nk] >= threshold:
                return True  # at least one neighbor
    return False  # completely isolated


def save_voxel_as_obj(filename, alpha, rgb, threshold=0.2, cube_size=1.0):
    """Save voxels with alpha > threshold as OBJ format, excluding completely isolated voxels."""
    with open(filename, 'w') as file:
        file.write("# OBJ file\n")

        # vertex offsets for each voxel cube
        cube_vertices = [
            (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)
        ]
        # face vertex indices
        cube_faces = [
            (1, 2, 3, 4), (5, 6, 7, 8), (1, 5, 8, 4),
            (2, 6, 7, 3), (1, 2, 6, 5), (4, 3, 7, 8)
        ]

        vertex_count = 1  # vertex counter

        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                for k in range(alpha.shape[2]):
                    if alpha[i, j, k] >= threshold :  #and has_adjacent_voxel(alpha, i, j, k, threshold):
                        # get color info
                        r, g, b = rgb[i, j, k]

                        # write vertices
                        for dx, dy, dz in cube_vertices:
                            x = i + dx * cube_size
                            y = j + dy * cube_size
                            z = k + dz * cube_size
                            file.write(f"v {x} {y} {z} {r} {g} {b}\n")

                        # write faces
                        for face in cube_faces:
                            v1, v2, v3, v4 = (vertex_count + idx - 1 for idx in face)
                            file.write(f"f {v1} {v2} {v3} {v4}\n")

                        # each cube has 8 vertices
                        vertex_count += 8

def save_voxel_as_uncertainty_obj(filename, alpha, uncertainty, threshold=0.85, cube_size=1.0):
    """Save voxels with alpha > threshold as OBJ format, with continuous color mapping based on uncertainty (white -> red)."""

    with open(filename, 'w') as file:
        file.write("# OBJ file (Continuous Uncertainty Visualization)\n")

        # find all voxel indices where alpha > threshold
        mask = alpha > threshold

        if np.any(mask):  # ensure valid data exists
            valid_uncertainty = uncertainty[mask]  # extract uncertainty values for valid voxels

            # normalize uncertainty to range 0-1
            min_uncertainty, max_uncertainty = np.min(valid_uncertainty), np.max(valid_uncertainty)
            normalized_uncertainty = (valid_uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty + 1e-8)  # avoid division by zero
            #breakpoint()
            # print uncertainty range
            print(f"Uncertainty Normalized Range: {min_uncertainty:.6f} ~ {max_uncertainty:.6f}")

        else:
            print("No valid voxels found above the threshold. OBJ file will not be generated.")
            return

        # color mapping: from white (1,1,1) to red (1,0,0) based on normalized_uncertainty
        def get_color(value):

            g_b_value = np.clip(1.0 - value * 10, 0.0, 1.0)
            return (1.0, g_b_value, g_b_value)  # R=1.0, G and B vary with uncertainty

        # vertex offsets for each voxel cube
        cube_vertices = [
            (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)
        ]
        # face vertex indices
        cube_faces = [
            (1, 2, 3, 4), (5, 6, 7, 8), (1, 5, 8, 4),
            (2, 6, 7, 3), (1, 2, 6, 5), (4, 3, 7, 8)
        ]

        vertex_count = 1  # OBJ vertex counter
        idx = 0  # index into uncertainty values

        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                for k in range(alpha.shape[2]):
                    if mask[i, j, k]:
                        # get color based on normalized uncertainty
                        r, g, b = get_color(normalized_uncertainty[idx])
                        idx += 1

                        # write vertices
                        for dx, dy, dz in cube_vertices:
                            x = i + dx * cube_size
                            y = j + dy * cube_size
                            z = k + dz * cube_size
                            file.write(f"v {x} {y} {z} {r} {g} {b}\n")

                        # write faces
                        for face in cube_faces:
                            v1, v2, v3, v4 = (vertex_count + idx - 1 for idx in face)
                            file.write(f"f {v1} {v2} {v3} {v4}\n")

                        # each cube has 8 vertices
                        vertex_count += 8


    plt.hist(valid_uncertainty, bins=20)
    plt.title("Uncertainty Value Distribution")
    plt.xlabel("Uncertainty Score")
    plt.ylabel("Voxel Count")

    plt.savefig("uncertainty_distribution.png", dpi=300, bbox_inches='tight')

    print(f"Uncertainty Voxel OBJ saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and process a .npz file.')
    parser.add_argument('--input_file', type=str, help='Path to the .npz file')
    parser.add_argument('--output_file', type=str, help='Path to the .obj file')
    parser.add_argument('--uncertainty_file', type=str, help='Path to the uncertainty .obj file')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    uncertainty_file = args.uncertainty_file  # uncertainty output filename


    # load data
    alpha, rgb, uncertainty = read_voxel_data(input_file)

    # save as .obj format
    save_voxel_as_obj(output_file, alpha, rgb)

    # # only save uncertainty file if uncertainty_file is not None
    # if args.uncertainty_file is not None:
    #     breakpoint()
    #     save_voxel_as_uncertainty_obj(args.uncertainty_file, alpha, uncertainty)
