"""
Demo gallery generator - v2 upgrade over generate_demo_images.py.

Replaces manual eyeballing of crop regions with an automatic scoring pass:
scans the held-out validation strip in a sliding window, scores each
candidate crop on building density, roof-material diversity, water
presence, and road-polyline jaggedness (turning-angle variance - a proxy
for "did the classical road detector zigzag here"), and surfaces the
top-N cleanest crops plus the worst ones (kept as a documented "known
failure" appendix instead of hidden).

For each selected crop, generates FOUR presentation formats:
  1. Side-by-side with in-image legend + live stats footer (self-contained,
     works even if screenshotted out of context)
  2. 2x Lanczos-upscaled version (sharper on a projector/big screen)
  3. Confidence heatmap variant (raw probability map, not just the final
     thresholded decision - shows judges the model's internal confidence)
  4. Animated GIF toggling input <-> output (punchier for a live demo)

Usage:
    python3 generate_demo_gallery.py --top 3 --show-rejects 2
"""
import argparse, os, json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
GALLERY_DIR = f'{OUT_DIR}/demo_gallery'

HELD_OUT_X_START = 800  # must match dataset.split_regions config used for the main model
CROP_WIDTH = 200        # full width of the held-out strip
CROP_HEIGHT = 220
Y_STRIDE = 30

ROOF_COLORS = {
    'RCC (Concrete)': (255, 80, 80),
    'Tiled': (255, 180, 60),
    'Tin/Metal Sheet': (120, 200, 255),
    'Other': (180, 180, 180),
}


def polyline_jaggedness(pts):
    """Std-dev of turning angle along a polyline, in radians. Near 0 =
    straight/smooth road. Large (>0.5-0.6) = visibly zigzagging."""
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    dirs = np.diff(pts, axis=0)
    angles = np.arctan2(dirs[:, 1], dirs[:, 0])
    turning = np.diff(angles)
    turning = (turning + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return float(np.std(turning))


def score_crop(y0, y1, x0, x1, buildings, roads, water):
    b_in = [b for b in buildings if y0 <= (b['bbox'][1] + b['bbox'][3] / 2) <= y1
            and x0 <= (b['bbox'][0] + b['bbox'][2] / 2) <= x1]
    w_in = [w for w in water if any(y0 <= p[1] <= y1 and x0 <= p[0] <= x1 for p in w['polygon_px'])]
    r_in = [r for r in roads if any(y0 <= p[1] <= y1 and x0 <= p[0] <= x1 for p in r['polyline_px'])]

    n_buildings = len(b_in)
    roof_types = set(b['roof_material'] for b in b_in)
    roof_diversity = len(roof_types)
    n_water = len(w_in)

    road_penalty = 0.0
    road_bonus = 0.0
    for r in r_in:
        j = polyline_jaggedness(r['polyline_px'])
        if j > 0.55:
            road_penalty += (j - 0.55) * 15  # heavily punish visible zigzag
        else:
            road_bonus += 2.0  # clean road segment is a nice visual

    score = (n_buildings * 1.0
             + roof_diversity * 6.0
             + n_water * 4.0
             + road_bonus
             - road_penalty)

    return {
        'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1,
        'score': round(score, 2),
        'n_buildings': n_buildings, 'roof_diversity': roof_diversity,
        'n_water': n_water, 'n_roads': len(r_in),
        'road_penalty': round(road_penalty, 2), 'road_bonus': round(road_bonus, 2),
    }


def get_font(size):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_annotated_sidebyside(rgb_crop, overlay_crop, stats, label, unseen, out_path):
    """Self-contained side-by-side: legend + stats footer baked into the
    image itself, so it still makes sense if screenshotted in isolation."""
    h, w = rgb_crop.shape[:2]
    pad_top, pad_bottom, gap, side_pad = 42, 78, 8, 14

    title_font = get_font(19)
    label_font = get_font(14)
    small_font = get_font(13)

    tag = 'held-out — unseen during training' if unseen else 'training region'
    title_text = f'{label}  ({tag})'

    # legend/stats entries, each (text, swatch_color_or_None)
    legend_items = [('RCC (Concrete)', ROOF_COLORS['RCC (Concrete)']),
                     ('Tiled', ROOF_COLORS['Tiled']),
                     ('Tin/Metal Sheet', ROOF_COLORS['Tin/Metal Sheet']),
                     ('Other', ROOF_COLORS['Other']),
                     ('Road', (245, 217, 10)),
                     ('Water', (0, 220, 220))]
    stats_text = (f"Buildings: {stats['n_buildings']}   |   Roof types shown: {stats['roof_diversity']}   |   "
                  f"Roads: {stats['n_roads']}   |   Water: {stats['n_water']}")

    # measure required widths using a scratch draw context
    scratch = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    title_w = scratch.textlength(title_text, font=title_font)
    stats_w = scratch.textlength(stats_text, font=small_font)
    legend_w = sum(22 + scratch.textlength(t, font=small_font) + 18 for t, _ in legend_items)

    image_row_w = w * 2 + gap
    content_w = max(image_row_w, title_w, stats_w, legend_w)
    canvas_w = int(content_w + side_pad * 2)
    canvas_h = h + pad_top + pad_bottom
    img_x_offset = side_pad + max(0, (content_w - image_row_w) // 2)

    canvas = Image.new('RGB', (canvas_w, canvas_h), (15, 20, 25))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(Image.fromarray(rgb_crop), (int(img_x_offset), pad_top))
    canvas.paste(Image.fromarray(overlay_crop), (int(img_x_offset + w + gap), pad_top))

    draw.text((canvas_w / 2, 8), title_text, font=title_font, fill='white', anchor='ma')
    draw.text((img_x_offset + w / 2, pad_top + h + 4), 'Input Orthophoto', font=label_font, fill='white', anchor='ma')
    draw.text((img_x_offset + w + gap + w / 2, pad_top + h + 4), 'AI-Extracted Features', font=label_font, fill='white', anchor='ma')

    legend_y = pad_top + h + 28
    lx = side_pad
    for text, color in legend_items:
        draw.rectangle([lx, legend_y, lx + 14, legend_y + 14], fill=color)
        draw.text((lx + 20, legend_y - 1), text, font=small_font, fill='white')
        lx += 20 + scratch.textlength(text, font=small_font) + 18

    stats_y = legend_y + 22
    draw.text((side_pad, stats_y), stats_text, font=small_font, fill=(180, 200, 220))

    canvas.save(out_path)


def make_upscaled(overlay_crop, rgb_crop, out_path, scale=2):
    """2x Lanczos upscale of the side-by-side pair for sharper projector display."""
    h, w = rgb_crop.shape[:2]
    combo = np.concatenate([rgb_crop, np.full((h, 6, 3), 255, dtype=np.uint8), overlay_crop], axis=1)
    up = cv2.resize(combo, (combo.shape[1] * scale, combo.shape[0] * scale), interpolation=cv2.INTER_LANCZOS4)
    Image.fromarray(up).save(out_path)


def make_heatmap_variant(rgb_crop, prob_crop, overlay_crop, out_path):
    """3-panel: RGB | raw confidence heatmap | final thresholded decision.
    Shows the model's internal probability, not just the binary output -
    good technical-depth slide, demonstrates you understand your own model's
    uncertainty rather than treating it as a black box."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    axes[0].imshow(rgb_crop); axes[0].set_title('Input', fontweight='bold', fontsize=11)
    im = axes[1].imshow(prob_crop, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Model Confidence (raw probability)', fontweight='bold', fontsize=11)
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('P(building)', fontsize=9)
    axes[2].imshow(overlay_crop); axes[2].set_title('Final Thresholded Output', fontweight='bold', fontsize=11)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def make_toggle_gif(rgb_crop, overlay_crop, out_path, duration_ms=1400):
    frames = [Image.fromarray(rgb_crop), Image.fromarray(overlay_crop)]
    frames[0].save(out_path, save_all=True, append_images=[frames[1]],
                    duration=duration_ms, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=3)
    parser.add_argument('--show-rejects', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(GALLERY_DIR, exist_ok=True)

    rgb = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB'))
    overlay = np.array(Image.open(f'{OUT_DIR}/best_extraction_overlay.png').convert('RGB'))
    prob = np.load(f'{OUT_DIR}/pred_prob_ensemble.npy')

    with open(f'{OUT_DIR}/buildings_best_enriched.json') as f:
        buildings = json.load(f)
    with open(f'{OUT_DIR}/roads_best.json') as f:
        roads = json.load(f)
    with open(f'{OUT_DIR}/water_best.json') as f:
        water = json.load(f)

    H, W = rgb.shape[:2]
    x0, x1 = HELD_OUT_X_START, W

    candidates = []
    for y0 in range(0, H - CROP_HEIGHT + 1, Y_STRIDE):
        y1 = y0 + CROP_HEIGHT
        candidates.append(score_crop(y0, y1, x0, x1, buildings, roads, water))

    candidates.sort(key=lambda c: c['score'], reverse=True)

    # Hard disqualification: a crop with visibly jagged roads should never
    # end up in the "recommended for demo" pool no matter how many
    # buildings it has - road_penalty existing at all means a judge could
    # spot a zigzag. Route those into the reject/failure-mode pool instead,
    # where they belong (documented on purpose, not accidentally shown off
    # as a good result).
    ROAD_PENALTY_DISQUALIFY = 8.0
    clean = [c for c in candidates if c['road_penalty'] <= ROAD_PENALTY_DISQUALIFY]
    flagged = [c for c in candidates if c['road_penalty'] > ROAD_PENALTY_DISQUALIFY]

    # de-overlap: greedily pick top CLEAN candidates that don't heavily overlap an already-picked one
    selected = []
    for c in clean:
        overlaps = any(not (c['y1'] <= s['y0'] or c['y0'] >= s['y1']) for s in selected)
        if not overlaps:
            selected.append(c)
        if len(selected) >= args.top:
            break

    # rejects: prefer flagged (visible failure mode) crops so the "known
    # issues" appendix is actually representative; fall back to worst-clean
    # if there aren't enough flagged ones
    rejects = sorted(flagged, key=lambda c: -c['road_penalty'])[:args.show_rejects]
    if len(rejects) < args.show_rejects:
        rejects += sorted(clean, key=lambda c: c['score'])[:args.show_rejects - len(rejects)]

    report_lines = ['=== Demo Crop Auto-Selection Report ===\n']
    report_lines.append(f'Scanned {len(candidates)} candidate windows (height={CROP_HEIGHT}px, stride={Y_STRIDE}px) '
                         f'within the held-out region (x>={HELD_OUT_X_START}).\n')

    for i, c in enumerate(selected, 1):
        name = f'gallery_{i}'
        label = f'Selected Region {i}'
        y0, y1 = c['y0'], c['y1']
        rgb_crop = rgb[y0:y1, x0:x1]
        overlay_crop = overlay[y0:y1, x0:x1]
        prob_crop = prob[y0:y1, x0:x1]

        make_annotated_sidebyside(rgb_crop, overlay_crop, c, label, True, f'{GALLERY_DIR}/{name}_annotated.png')
        make_upscaled(overlay_crop, rgb_crop, f'{GALLERY_DIR}/{name}_upscaled2x.png')
        make_heatmap_variant(rgb_crop, prob_crop, overlay_crop, f'{GALLERY_DIR}/{name}_heatmap.png')
        make_toggle_gif(rgb_crop, overlay_crop, f'{GALLERY_DIR}/{name}_toggle.gif')

        report_lines.append(f'#{i} y=[{y0},{y1}] x=[{x0},{x1}]  score={c["score"]}  '
                             f'(buildings={c["n_buildings"]}, roofs={c["roof_diversity"]}, '
                             f'water={c["n_water"]}, roads={c["n_roads"]}, road_penalty={c["road_penalty"]})')
        print(f'Generated {name}: score={c["score"]:.1f} (4 formats: annotated, upscaled, heatmap, gif)')

    report_lines.append('\n=== Rejected (worst-scoring) crops, kept for honest Q&A prep ===')
    for i, c in enumerate(rejects, 1):
        name = f'reject_{i}'
        y0, y1 = c['y0'], c['y1']
        rgb_crop = rgb[y0:y1, x0:x1]
        overlay_crop = overlay[y0:y1, x0:x1]
        make_annotated_sidebyside(rgb_crop, overlay_crop, c, f'Reject Example {i} (low score)', True,
                                   f'{GALLERY_DIR}/{name}_annotated.png')
        reason = f'road jaggedness too high (penalty={c["road_penalty"]:.1f} > threshold {ROAD_PENALTY_DISQUALIFY})' \
                 if c['road_penalty'] > ROAD_PENALTY_DISQUALIFY else 'sparse/empty region (low overall score)'
        report_lines.append(f'#{i} y=[{y0},{y1}] x=[{x0},{x1}]  score={c["score"]}  reason: {reason}')
        print(f'Generated {name}: score={c["score"]:.1f} (documented reject, reason: {reason})')

    with open(f'{GALLERY_DIR}/selection_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    print(f'\nFull report: {GALLERY_DIR}/selection_report.txt')


if __name__ == '__main__':
    main()
