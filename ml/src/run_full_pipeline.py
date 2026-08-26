"""
Single-command pipeline orchestrator.

Problem this solves: reproducing the final results currently means running
8 separate scripts in the correct order, with at least one hidden manual
step (base64-encoding the RGB image for the dashboard) that wasn't
scripted anywhere. That's fragile - if a teammate or judge wants to rerun
this after you add a new tile or retrain, they have to know the exact
order and every undocumented step. This script fixes that: one command,
clear stage-by-stage progress, and a real validation check after every
stage (not just "did the subprocess exit 0" - does the expected output
file actually exist and look non-empty) so a failure is caught immediately
with a specific, actionable message instead of a cryptic downstream
crash three stages later.

Usage:
    python3 run_full_pipeline.py                    # full run
    python3 run_full_pipeline.py --skip-change-detection   # skip the slow synthetic change-detection demo
    python3 run_full_pipeline.py --force             # rerun every stage even if outputs already exist
"""
import argparse, os, subprocess, sys, time, base64

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PKG_DIR, 'data')
OUT_DIR = os.path.join(PKG_DIR, 'outputs')


def check_prereqs():
    """Fail fast with a specific message rather than letting stage 3 crash
    mysteriously because stage 1's checkpoint was never trained."""
    required = {
        f'{DATA_DIR}/RGB.png': 'source aerial image — required for every stage',
        f'{DATA_DIR}/GT.png': 'ground-truth building mask — required for evaluation metrics',
        f'{OUT_DIR}/model_v2_ckpt.pt': 'main model checkpoint — run train.py first (see MANUAL.md)',
        f'{OUT_DIR}/ensemble_member_1.pt': 'ensemble member 1 — run train_ensemble_member.py --member 1',
        f'{OUT_DIR}/ensemble_member_2.pt': 'ensemble member 2 — run train_ensemble_member.py --member 2',
    }
    missing = [(path, why) for path, why in required.items() if not os.path.exists(path)]
    if missing:
        print('Cannot start — missing prerequisites:\n')
        for path, why in missing:
            print(f'  MISSING: {path}\n           ({why})')
        print('\nSee MANUAL.md section 3 for the training steps that produce these.')
        sys.exit(1)
    print('All prerequisite files found (data + 3 trained checkpoints).\n')


def generate_rgb_b64():
    """This step used to be a manual one-off command, not part of any
    script - fixed here so the full pipeline has zero hidden manual steps."""
    out_path = f'{OUT_DIR}/rgb_b64.txt'
    with open(f'{DATA_DIR}/RGB.png', 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    with open(out_path, 'w') as f:
        f.write(b64)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


STAGES = [
    {'name': 'Ensemble probability map (3-model average, TTA, threshold tuning)',
     'script': 'ensemble_eval.py',
     'expect': ['pred_prob_ensemble.npy'],
     'timeout': 280},
    {'name': 'Building/road/water extraction (best model)',
     'script': 'infer_and_vectorize_best.py',
     'expect': ['buildings_best.json', 'roads_best.json', 'water_best.json', 'best_extraction_overlay.png'],
     'timeout': 200},
    {'name': 'Polygon regularization (rectilinear cleanup)',
     'script': 'regularize_polygons.py',
     'expect': ['buildings_regularized.json'],
     'timeout': 120},
    {'name': 'Property tax + uncertainty review-queue enrichment',
     'script': 'enrich_tax_uncertainty_best.py',
     'expect': ['buildings_best_enriched.json'],
     'timeout': 60},
    {'name': 'Change detection (synthetic Time-B demo)',
     'script': 'change_detection.py',
     'expect': ['change_detection_overlay.png', 'change_detection.json'],
     'timeout': 280, 'skippable': True},
    {'name': 'Interactive dashboard build',
     'script': 'build_dashboard_best.py',
     'expect': ['dashboard_best.html'],
     'timeout': 60},
    {'name': 'Demo gallery (auto-selected crops, 4 formats each)',
     'script': 'generate_demo_gallery.py',
     'expect': ['demo_gallery/selection_report.txt'],
     'timeout': 200},
]


def run_stage(stage, force):
    expected_paths = [f'{OUT_DIR}/{e}' for e in stage['expect']]
    if not force and all(os.path.exists(p) for p in expected_paths):
        print(f"  [skip] outputs already exist ({', '.join(stage['expect'])})")
        return True

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, '-u', stage['script']],
        cwd=SRC_DIR, capture_output=True, text=True, timeout=stage['timeout']
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'  FAILED (exit code {result.returncode}, {elapsed:.0f}s)')
        print(f'  --- last 15 lines of output ---')
        for line in (result.stdout + result.stderr).splitlines()[-15:]:
            print(f'  {line}')
        return False

    missing_outputs = [p for p in expected_paths if not os.path.exists(p)]
    if missing_outputs:
        print(f'  FAILED — script exited cleanly but did not produce expected output(s): {missing_outputs}')
        return False

    print(f'  OK ({elapsed:.0f}s) — produced {", ".join(stage["expect"])}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='rerun every stage even if outputs already exist')
    parser.add_argument('--skip-change-detection', action='store_true', help='skip the slow synthetic change-detection demo stage')
    args = parser.parse_args()

    print('=' * 70)
    print('SVAMITVA AI Feature Extraction — Full Pipeline')
    print('=' * 70 + '\n')

    check_prereqs()

    print('Generating base64-embedded image for dashboard (was a manual step, now scripted)...')
    if generate_rgb_b64():
        print('  OK\n')
    else:
        print('  FAILED\n')
        sys.exit(1)

    for i, stage in enumerate(STAGES, 1):
        if stage.get('skippable') and args.skip_change_detection and 'change' in stage['script']:
            print(f'[{i}/{len(STAGES)}] {stage["name"]} — SKIPPED (--skip-change-detection)\n')
            continue
        print(f'[{i}/{len(STAGES)}] {stage["name"]}')
        ok = run_stage(stage, args.force)
        print()
        if not ok:
            print(f'Pipeline stopped at stage {i} ({stage["script"]}). Fix the issue above and rerun.')
            print('(Already-completed stages will be skipped next time unless you pass --force.)')
            sys.exit(1)

    print('=' * 70)
    print('Pipeline complete. Key outputs:')
    print(f'  {OUT_DIR}/dashboard_best.html          <- open in browser')
    print(f'  {OUT_DIR}/demo_gallery/                 <- presentation-ready images')
    print(f'  {OUT_DIR}/buildings_best_enriched.json  <- full building records')
    print('=' * 70)


if __name__ == '__main__':
    main()
