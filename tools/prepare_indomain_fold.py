"""Build the in-domain quadrant split (fold ID) of the 2023-02-17 UMBRA-04 collect.

Same-acquisition capability measurement: train/val/test are separate QUADRANT
FILES of one collect, so crops can never straddle splits (leakage-proof by
construction, no buffer logic needed).

  train: 42023-*top_right + 12023-*bottom_left   (~896+420 aircraft)
  val:   32023-*top_left                          (~204)  - best-epoch selection
  test:  22023-*bottom_right                      (~235)  - reported, untouched

Images are symlinked and annfiles copied from fold_0/train (every 2023-02-17
crop is in every CV fold's train set, filenames = crop_{crop_idx:04d}).
"""
import json
import os
import shutil
from pathlib import Path

POOL_META = Path(os.environ.get(
    'UMBRA_POOL_METADATA', 'datasets/Umbra/pool_metadata.json'))
SRC = Path(os.environ.get(
    'UMBRA_CV_TRAIN_ROOT', 'datasets/Umbra/umbra_cv/fold_0/train'))
DST = Path(os.environ.get(
    'UMBRA_INDOMAIN_ROOT', 'datasets/Umbra/umbra_indomain'))

SPLITS = {
    'train': ['42023-02-17-17-04-17_UMBRA-04_top_right.tif',
              '12023-02-17-17-04-17_UMBRA-04_bottom_left.tif'],
    'val':   ['32023-02-17-17-04-17_UMBRA-04_top_left.tif'],
    'test':  ['22023-02-17-17-04-17_UMBRA-04_bottom_right.tif'],
}

meta = json.loads(POOL_META.read_text())
by_split = {s: [] for s in SPLITS}
src2split = {img: s for s, imgs in SPLITS.items() for img in imgs}

for e in meta:
    s = src2split.get(e['source_image'])
    if s:
        by_split[s].append(e['crop_idx'])

report = {}
for split, idxs in by_split.items():
    img_d = DST / split / 'images'
    ann_d = DST / split / 'annfiles'
    img_d.mkdir(parents=True, exist_ok=True)
    ann_d.mkdir(parents=True, exist_ok=True)
    missing = 0
    for idx in idxs:
        name = f'crop_{idx:04d}'
        src_img = SRC / 'images' / f'{name}.png'
        src_ann = SRC / 'annfiles' / f'{name}.txt'
        if not src_img.exists() or not src_ann.exists():
            missing += 1
            continue
        dst_img = img_d / f'{name}.png'
        if not dst_img.exists():
            dst_img.symlink_to(src_img)
        shutil.copy2(src_ann, ann_d / f'{name}.txt')
    n_air = sum(1 for f in ann_d.glob('*.txt') for _ in open(f))
    report[split] = dict(crops=len(idxs), missing=missing,
                         linked=len(list(img_d.glob('*.png'))), gt_lines=n_air)
    print(split, report[split])

(DST / 'split_report.json').write_text(json.dumps(
    {'splits': {k: v for k, v in SPLITS.items()}, 'counts': report}, indent=1))
print('done ->', DST)
