import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from files_utils import load_hex, collect


def plot_palette(palette, name, save_path=None):
    fig, ax = plt.subplots(figsize=(len(palette), 2))
    for i, color in enumerate(palette):
        rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor='none',
                                 facecolor=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')
        ax.add_patch(rect)
    plt.xlim(0, len(palette))
    plt.ylim(0, 1)
    ax.axis('off')
    plt.title(f'Palette: {name}')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def display_palette(hex_path, save_dir=None):
    palette = load_hex(hex_path)
    name = os.path.splitext(os.path.basename(hex_path))[0]
    save_path = os.path.join(save_dir, f"{name}.png") if save_dir else None
    plot_palette(palette, name, save_path=save_path)


def display_palettes(hex_dir, save_dir=None):
    palette_paths = collect(hex_dir, '.hex')
    for path in palette_paths:
        path = ''.join(path)  # flatten if needed
        display_palette(path, save_dir)


def main():
    parser = argparse.ArgumentParser(description='Visualize .hex color palettes')
    parser.add_argument('--hex_path', type=str, default=None, help='Path to a single .hex file')
    parser.add_argument('--hex_dir', type=str, default=None, help='Path to a directory of .hex files')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save visualized images')

    args = parser.parse_args()

    if args.hex_path:
        display_palette(args.hex_path, save_dir=args.save_dir)
    elif args.hex_dir:
        display_palettes(args.hex_dir, save_dir=args.save_dir)
    else:
        print('❌ Please provide either --hex_path or --hex_dir')


if __name__ == '__main__':
    main()
