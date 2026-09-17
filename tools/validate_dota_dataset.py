#!/usr/bin/env python3
"""Validate image/annotation pairing and DOTA quadrilateral records."""

import argparse
import math
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def validate_split(root: Path, strict_bounds: bool) -> tuple[list[str], int]:
    images_dir = root / 'images'
    ann_dir = root / 'annfiles'
    errors: list[str] = []
    outside_count = 0
    if not images_dir.is_dir() or not ann_dir.is_dir():
        return [f'{root}: expected images/ and annfiles/ directories'], 0

    images = {
        path.stem: path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    annotations = {path.stem: path for path in ann_dir.glob('*.txt')}

    for stem in sorted(images.keys() - annotations.keys()):
        errors.append(f'{root}: image has no annotation file: {images[stem].name}')
    for stem in sorted(annotations.keys() - images.keys()):
        errors.append(f'{root}: annotation has no image: {annotations[stem].name}')

    object_count = 0
    for stem in sorted(images.keys() & annotations.keys()):
        try:
            with Image.open(images[stem]) as image:
                width, height = image.size
        except Exception as exc:
            errors.append(f'{images[stem]}: cannot read image: {exc}')
            continue

        for line_number, raw in enumerate(
                annotations[stem].read_text().splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 10:
                errors.append(
                    f'{annotations[stem]}:{line_number}: expected 10 fields, '
                    f'found {len(parts)}')
                continue
            try:
                coords = [float(value) for value in parts[:8]]
                difficulty = int(parts[9])
            except ValueError:
                errors.append(
                    f'{annotations[stem]}:{line_number}: invalid numeric field')
                continue
            if not all(math.isfinite(value) for value in coords):
                errors.append(
                    f'{annotations[stem]}:{line_number}: non-finite coordinate')
            xs, ys = coords[0::2], coords[1::2]
            if min(xs) < 0 or max(xs) > width or min(ys) < 0 or max(ys) > height:
                outside_count += 1
                if strict_bounds:
                    errors.append(
                        f'{annotations[stem]}:{line_number}: box is outside '
                        f'{width}x{height} image bounds')
            if parts[8] != 'aircraft':
                errors.append(
                    f'{annotations[stem]}:{line_number}: expected class '
                    f'"aircraft", found "{parts[8]}"')
            if difficulty not in (0, 1):
                errors.append(
                    f'{annotations[stem]}:{line_number}: difficulty must be 0 or 1')
            object_count += 1

    print(f'{root}: {len(images)} images, {len(annotations)} annotations, '
          f'{object_count} objects')
    return errors, outside_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', type=Path,
        help='a split containing images/ and annfiles/, or a fold root')
    parser.add_argument(
        '--strict-bounds', action='store_true',
        help='treat partly out-of-frame boxes as errors instead of warnings')
    args = parser.parse_args()

    roots = ([args.root] if (args.root / 'images').is_dir() else
             [path for name in ('train', 'val', 'test')
              if (path := args.root / name).is_dir()])
    if not roots:
        print(f'{args.root}: no dataset split found')
        return 2

    errors: list[str] = []
    outside_count = 0
    for root in roots:
        split_errors, split_outside = validate_split(root, args.strict_bounds)
        errors.extend(split_errors)
        outside_count += split_outside
    for error in errors:
        print(f'ERROR: {error}')
    if outside_count and not args.strict_bounds:
        print(f'WARNING: {outside_count} box(es) extend outside image bounds; '
              'use --strict-bounds to reject them')
    if errors:
        print(f'FAILED: {len(errors)} problem(s)')
        return 1
    print('OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
