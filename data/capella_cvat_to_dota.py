"""Convert the Capella CVAT annotations to DOTA-format 512x512 chips.

Source: GrokSAR/datasets/SAR-AIRcraft-1.0/capella_aircraft/<scene>/annotations.xml
        + the scene's *.tif image(s) (8-bit dB-scaled, GEO products).

Facts this script encodes (verified 2026-07-14):
- ALL Capella annotations are horizontal boxes (xtl/ytl/xbr/ybr); there are
  no polygons. The Capella tier is therefore an HBB evaluation. Boxes are
  written as axis-aligned 8-coord DOTA polygons so the same DOTA tooling
  (and DotaBig2SmallMetric) can read them.
- Labels: airplane variants (incl. dmg/*) map to class 'aircraft';
  helicopters are EXCLUDED (the Umbra training pool has no helicopter
  class, so counting them as aircraft would corrupt cross-sensor eval).
- Scenes must be grouped by SITE for any split (6 Khartoum scenes = 1 site).

Output layout (test-tier, whole set):
  <out>/images/<scene>__<imgstem>__x<X>_y<Y>.png
  <out>/annfiles/<same>.txt          # DOTA: 8 coords + 'aircraft 0'
  <out>/sites.json                   # scene -> site manifest + counts
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

CHIP = 512
STRIDE = 384
MIN_KEEP_FRAC = 0.5      # keep a clipped box if >= this fraction remains
HELI_RE = re.compile(r'helicopter', re.IGNORECASE)


def site_of(scene: str) -> str:
    s = scene.lower()
    if 'khartoum' in s:
        return 'khartoum'
    return re.split(r'[_-]\d', scene)[0].rstrip('_-')


def find_tif(scene_dir: Path, name: str):
    """Locate the tif for a CVAT <image name=...> anywhere under the scene."""
    cands = [p for p in scene_dir.rglob('*.tif') if p.name == name]
    if cands:
        return cands[0]
    # CVAT task name sometimes lacks suffixes the file gained (or vice
    # versa) — fall back to stem-prefix matching.
    stem = Path(name).stem
    cands = [p for p in scene_dir.rglob('*.tif')
             if p.stem.startswith(stem) or stem.startswith(p.stem)]
    return cands[0] if len(cands) == 1 else None


def parse_scene(xml_path: Path):
    """Yield (image_name, [ (xtl, ytl, xbr, ybr, label), ... ])."""
    root = ET.parse(xml_path).getroot()
    for img in root.iter('image'):
        boxes = []
        for b in img.iter('box'):
            boxes.append((
                float(b.get('xtl')), float(b.get('ytl')),
                float(b.get('xbr')), float(b.get('ybr')),
                b.get('label', '')))
        yield img.get('name'), boxes


def chip_scene(arr, boxes, scene, stem, out_img, out_ann):
    """Tile one image; keep chips containing >= 1 box center."""
    H, W = arr.shape[:2]
    n_chips = n_boxes = 0
    xs = list(range(0, max(W - CHIP, 0) + 1, STRIDE)) or [0]
    ys = list(range(0, max(H - CHIP, 0) + 1, STRIDE)) or [0]
    if xs[-1] + CHIP < W:
        xs.append(W - CHIP)
    if ys[-1] + CHIP < H:
        ys.append(H - CHIP)
    for y0 in ys:
        for x0 in xs:
            x1, y1 = x0 + CHIP, y0 + CHIP
            lines = []
            for (bx0, by0, bx1, by1, _label) in boxes:
                cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
                if not (x0 <= cx < x1 and y0 <= cy < y1):
                    continue
                # clip to chip
                cx0, cy0 = max(bx0, x0), max(by0, y0)
                cx1, cy1 = min(bx1, x1), min(by1, y1)
                area = max(cx1 - cx0, 0) * max(cy1 - cy0, 0)
                orig = max(bx1 - bx0, 0) * max(by1 - by0, 0)
                if orig <= 0 or area / orig < MIN_KEEP_FRAC:
                    continue
                a, b, c, d = cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0
                lines.append(
                    f'{a:.1f} {b:.1f} {c:.1f} {b:.1f} '
                    f'{c:.1f} {d:.1f} {a:.1f} {d:.1f} aircraft 0')
            if not lines:
                continue
            name = f'{scene}__{stem}__x{x0}_y{y0}'
            chip = arr[y0:y1, x0:x1]
            if chip.shape[0] < CHIP or chip.shape[1] < CHIP:
                pad = np.zeros((CHIP, CHIP), dtype=chip.dtype)
                pad[:chip.shape[0], :chip.shape[1]] = chip
                chip = pad
            Image.fromarray(chip).save(out_img / f'{name}.png')
            (out_ann / f'{name}.txt').write_text('\n'.join(lines) + '\n')
            n_chips += 1
            n_boxes += len(lines)
    return n_chips, n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True,
                    help='CVAT-export scene directory (not included)')
    ap.add_argument('--out', default='datasets/capella_test',
                    help='output directory for generated DOTA chips')
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out_img, out_ann = out / 'images', out / 'annfiles'
    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    manifest, skipped = {}, []
    for xml_path in sorted(src.glob('*/annotations.xml')):
        scene_dir = xml_path.parent
        scene = scene_dir.name
        stats = dict(site=site_of(scene), images=0, chips=0,
                     aircraft=0, helicopters_excluded=0)
        for img_name, boxes in parse_scene(xml_path):
            keep = [b for b in boxes if not HELI_RE.search(b[4])]
            stats['helicopters_excluded'] += len(boxes) - len(keep)
            if not keep:
                continue
            tif = find_tif(scene_dir, img_name)
            if tif is None:
                skipped.append((scene, img_name))
                continue
            arr = np.asarray(Image.open(tif).convert('L'))
            n_c, n_b = chip_scene(arr, keep, scene, Path(img_name).stem,
                                  out_img, out_ann)
            stats['images'] += 1
            stats['chips'] += n_c
            stats['aircraft'] += n_b
        manifest[scene] = stats
        print(f"{scene:<38} site={stats['site']:<18} imgs={stats['images']} "
              f"chips={stats['chips']:>3} aircraft={stats['aircraft']:>4} "
              f"heli-excl={stats['helicopters_excluded']}")

    (out / 'sites.json').write_text(json.dumps(manifest, indent=2))
    total_chips = sum(s['chips'] for s in manifest.values())
    total_air = sum(s['aircraft'] for s in manifest.values())
    n_sites = len({s['site'] for s in manifest.values()})
    print(f'\nTOTAL: {total_chips} chips, {total_air} aircraft boxes, '
          f'{len(manifest)} scenes, {n_sites} sites')
    if skipped:
        print('SKIPPED (no tif matched):')
        for s in skipped:
            print('  ', s)


if __name__ == '__main__':
    main()
