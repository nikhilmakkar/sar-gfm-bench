#!/usr/bin/env python3
"""Create leakage-resistant DOTA folds from user-labelled image chips."""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    if destination.exists() or destination.is_symlink():
        return
    if copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Group DOTA chips by acquisition and build CV folds.')
    parser.add_argument('source', type=Path,
                        help='DOTA directory with images/ and annfiles/')
    parser.add_argument('groups_csv', type=Path,
                        help='CSV with columns stem,acquisition')
    parser.add_argument('output', type=Path)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--always-train', action='append', default=[],
                        help='acquisition excluded from validation; repeatable')
    parser.add_argument('--copy', action='store_true',
                        help='copy files instead of creating symlinks')
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty to prevent stale split files')

    image_dir, ann_dir = args.source / 'images', args.source / 'annfiles'
    if not image_dir.is_dir() or not ann_dir.is_dir():
        parser.error('source must contain images/ and annfiles/')
    if args.folds < 2:
        parser.error('--folds must be at least 2')

    images = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in images:
                parser.error(f'duplicate image stem: {path.stem}')
            images[path.stem] = path

    stem_to_group = {}
    with args.groups_csv.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {'stem', 'acquisition'} <= set(reader.fieldnames):
            parser.error('groups CSV must have stem,acquisition columns')
        for row in reader:
            stem_to_group[row['stem']] = row['acquisition']

    missing = sorted(set(images) - set(stem_to_group))
    extra = sorted(set(stem_to_group) - set(images))
    if missing or extra:
        parser.error(
            f'groups CSV mismatch: {len(missing)} image stems missing, '
            f'{len(extra)} unknown stems')

    grouped = defaultdict(list)
    group_objects = defaultdict(int)
    for stem, image in images.items():
        annotation = ann_dir / f'{stem}.txt'
        if not annotation.is_file():
            parser.error(f'missing annotation: {annotation}')
        group = stem_to_group[stem]
        grouped[group].append((image, annotation))
        group_objects[group] += sum(
            bool(line.strip()) for line in annotation.read_text().splitlines())

    always = set(args.always_train)
    unknown_always = always - set(grouped)
    if unknown_always:
        parser.error(f'unknown --always-train groups: {sorted(unknown_always)}')

    assignments = [[] for _ in range(args.folds)]
    totals = [0] * args.folds
    candidates = sorted(
        (group for group in grouped if group not in always),
        key=lambda group: (-group_objects[group], group))
    if len(candidates) < args.folds:
        parser.error('fewer validation-eligible acquisitions than folds')
    for group in candidates:
        fold = min(range(args.folds), key=lambda index: totals[index])
        assignments[fold].append(group)
        totals[fold] += group_objects[group]

    for fold, val_groups in enumerate(assignments):
        val_group_set = set(val_groups)
        for split in ('train', 'val'):
            (args.output / f'fold_{fold}' / split / 'images').mkdir(
                parents=True, exist_ok=True)
            (args.output / f'fold_{fold}' / split / 'annfiles').mkdir(
                parents=True, exist_ok=True)
        for group, pairs in grouped.items():
            split = 'val' if group in val_group_set else 'train'
            split_root = args.output / f'fold_{fold}' / split
            for image, annotation in pairs:
                link_or_copy(image, split_root / 'images' / image.name, args.copy)
                link_or_copy(annotation,
                             split_root / 'annfiles' / annotation.name,
                             args.copy)
        print(f'fold {fold}: val={val_groups}; val objects={totals[fold]}')

    print(f'created {args.folds} folds under {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
