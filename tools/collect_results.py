"""Aggregate per-fold results into the benchmark comparison table.

Parses mmengine JSON logs (work_dirs/<run>/<timestamp>/vis_data/scalars.json)
for the best dota/mAP per run, groups runs by backbone tag across folds, and
prints per-fold AP50 + mean/std alongside the historical DenoDet baselines.

Usage: python tools/collect_results.py [--work-dir work_dirs]
"""

import argparse
import json
import re
from pathlib import Path

# Historical reference rows (DenoDet ResNet-18 + H2RBox, CV_RESULTS_SUMMARY.md)
REFERENCE = {
    'denodet-r18 (2026-02)': [0.467, 0.171, 0.007, 0.608],
    'denodet-r18 +targeted-synth (2026-03)': [0.534, 0.300, 0.014, 0.765],
}

RUN_RE = re.compile(r'^orcnn_(?P<tag>.+)_fold(?P<fold>\d)$')


def best_map(run_dir: Path):
    best = None
    for scalars in run_dir.glob('*/vis_data/scalars.json'):
        for line in scalars.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = rec.get('dota/mAP')
            if v is not None and (best is None or v > best):
                best = v
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', default='work_dirs')
    args = parser.parse_args()

    runs = {}
    for d in sorted(Path(args.work_dir).iterdir()):
        m = RUN_RE.match(d.name)
        if not (d.is_dir() and m):
            continue
        runs.setdefault(m['tag'], {})[int(m['fold'])] = best_map(d)

    rows = dict(REFERENCE)
    for tag, folds in runs.items():
        rows[tag] = [folds.get(i) for i in range(4)]

    print(f"\n{'run':<42}{'f0':>8}{'f1':>8}{'f2':>8}{'f3':>8}"
          f"{'mean':>8}{'std':>7}")
    print('-' * 89)
    for name, vals in rows.items():
        cells = ''.join(
            f'{v:>8.3f}' if isinstance(v, float) else f'{"—":>8}'
            for v in vals)
        done = [v for v in vals if isinstance(v, float)]
        if len(done) == 4:
            mean = sum(done) / 4
            std = (sum((v - mean) ** 2 for v in done) / 4) ** 0.5
            print(f'{name:<42}{cells}{mean:>8.3f}{std:>7.3f}')
        else:
            print(f'{name:<42}{cells}{"—":>8}{"—":>7}')


if __name__ == '__main__':
    main()
