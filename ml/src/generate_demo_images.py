"""
Generates presentation-ready demo images: input orthophoto vs AI-extracted
features, side-by-side, for a chosen crop region of the tile.

By default, crops are taken from the HELD-OUT validation strip (x >= 800)
so they are honestly "unseen during training" - meaningful for a live demo
where you want to show the model working on data it wasn't fit to, not
just re-displaying the training region.

Usage:
    python3 generate_demo_images.py
        -> regenerates the default 3 demo crops used in the presentation

    python3 generate_demo_images.py --custom 500 750 800 1000 --name my_crop --label "My region"
        -> generates one additional crop at y=[500,750], x=[800,1000]

Requires outputs/best_extraction_overlay.png and data/RGB.png to already
exist (i.e. infer_and_vectorize_best.py has been run at least once).
"""
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
DEMO_DIR = f'{OUT_DIR}/demo_images'

# Held-out validation region boundary (must match dataset.split_regions
# with val_fraction=0.2, axis='x', side='right' - the config used for the
# main trained model). Crops with x0 >= this value are genuinely unseen.
HELD_OUT_X_START = 800

DEFAULT_CROPS = [
    {'name': 'demo_1_top', 'y0': 0, 'y1': 320, 'x0': 800, 'x1': 1000,
     'label': 'Industrial/commercial cluster'},
    {'name': 'demo_2_middle', 'y0': 380, 'y1': 700, 'x0': 800, 'x1': 1000,
     'label': 'Mixed residential + waterbodies'},
    {'name': 'demo_3_bottom', 'y0': 500, 'y1': 750, 'x0': 800, 'x1': 1000,
     'label': 'Dense residential cluster'},
]


def generate_crop(rgb, overlay, y0, y1, x0, x1, name, label):
    import os
    os.makedirs(DEMO_DIR, exist_ok=True)

    rgb_crop = rgb[y0:y1, x0:x1]
    overlay_crop = overlay[y0:y1, x0:x1]
    h, w = rgb_crop.shape[:2]
    aspect = w / h

    is_unseen = x0 >= HELD_OUT_X_START
    tag = 'held-out region — not seen during training' if is_unseen else \
          'WARNING: overlaps training region — not a fair "unseen data" demo'

    fig_h = 6
    fig_w = fig_h * aspect * 2 * 1.05
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), gridspec_kw={'wspace': 0.03})
    axes[0].imshow(rgb_crop)
    axes[0].set_title('Input: Drone Orthophoto', fontweight='bold', fontsize=13)
    axes[1].imshow(overlay_crop)
    axes[1].set_title('AI-Extracted Features', fontweight='bold', fontsize=13)
    for ax in axes:
        ax.axis('off')
    fig.suptitle(f'{label}  ({tag})', fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(f'{DEMO_DIR}/{name}_sidebyside.png', dpi=150, bbox_inches='tight')
    plt.close()

    Image.fromarray(rgb_crop).save(f'{DEMO_DIR}/{name}_input.png')
    Image.fromarray(overlay_crop).save(f'{DEMO_DIR}/{name}_output.png')

    print(f'Saved {name} (unseen={is_unseen}): {DEMO_DIR}/{name}_sidebyside.png')
    if not is_unseen:
        print(f'  {tag}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom', nargs=4, type=int, metavar=('Y0', 'Y1', 'X0', 'X1'),
                         help='generate one custom crop instead of the default 3')
    parser.add_argument('--name', type=str, default='custom_crop')
    parser.add_argument('--label', type=str, default='Custom region')
    args = parser.parse_args()

    rgb = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB'))
    overlay = np.array(Image.open(f'{OUT_DIR}/best_extraction_overlay.png').convert('RGB'))

    if args.custom:
        y0, y1, x0, x1 = args.custom
        generate_crop(rgb, overlay, y0, y1, x0, x1, args.name, args.label)
    else:
        for c in DEFAULT_CROPS:
            generate_crop(rgb, overlay, c['y0'], c['y1'], c['x0'], c['x1'], c['name'], c['label'])

    print(f'\nAll demo images in {DEMO_DIR}/')


if __name__ == '__main__':
    main()
