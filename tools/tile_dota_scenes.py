#!/usr/bin/env python3
"""Tile full SAR scenes and full-scene DOTA labels into runnable DOTA chips."""

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, length - tile_size + 1, stride))
    if values[-1] != length - tile_size:
        values.append(length - tile_size)
    return values


def parse_dota(path: Path) -> list[tuple[np.ndarray, str, int]]:
    records = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 10:
            raise ValueError(f'{path}:{line_number}: expected 10 fields')
        points = np.asarray([float(value) for value in parts[:8]],
                            dtype=np.float32).reshape(4, 2)
        records.append((points, parts[8], int(parts[9])))
    return records


def to_uint8(array: np.ndarray, mode: str, lower: float, upper: float) -> np.ndarray:
    if array.ndim == 3:
        array = array[..., 0]
    if mode == 'none':
        if array.dtype != np.uint8:
            raise ValueError('--intensity none requires an 8-bit input image')
        return array
    values = array.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    values = np.where(finite, values, 0)
    if mode == 'log-percentile':
        values = np.log1p(np.maximum(values, 0))
    if mode in {'log-percentile', 'percentile'}:
        lo, hi = np.percentile(values[finite], (lower, upper))
    else:
        lo, hi = float(values[finite].min()), float(values[finite].max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def clipped_box(points: np.ndarray, x: int, y: int, size: int,
                min_visible: float) -> np.ndarray | None:
    rectangle = np.asarray(
        [[x, y], [x + size, y], [x + size, y + size], [x, y + size]],
        dtype=np.float32)
    object_area = abs(float(cv2.contourArea(points)))
    if object_area <= 0:
        return None
    visible_area, intersection = cv2.intersectConvexConvex(points, rectangle)
    if intersection is None or visible_area / object_area < min_visible:
        return None
    local = intersection.reshape(-1, 2) - np.asarray([x, y], dtype=np.float32)
    return cv2.boxPoints(cv2.minAreaRect(local)).astype(np.float32)


def keep_empty(stem: str, x: int, y: int, ratio: float, seed: int) -> bool:
    digest = hashlib.sha256(f'{seed}:{stem}:{x}:{y}'.encode()).digest()
    value = int.from_bytes(digest[:8], 'big') / (2**64 - 1)
    return value < ratio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=Path,
                        help='directory containing full-scene images')
    parser.add_argument('annotations', type=Path,
                        help='directory containing same-stem full-scene DOTA files')
    parser.add_argument('output', type=Path,
                        help='new DOTA chip dataset directory')
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--min-visible', type=float, default=0.7)
    parser.add_argument('--empty-ratio', type=float, default=0.1,
                        help='deterministic fraction of empty chips to retain')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--intensity', choices=('log-percentile', 'percentile', 'minmax', 'none'),
        default='log-percentile')
    parser.add_argument('--lower-percentile', type=float, default=1.0)
    parser.add_argument('--upper-percentile', type=float, default=99.0)
    args = parser.parse_args()

    if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size:
        parser.error('require tile-size > 0 and 0 <= overlap < tile-size')
    if not 0 <= args.min_visible <= 1 or not 0 <= args.empty_ratio <= 1:
        parser.error('min-visible and empty-ratio must be between 0 and 1')
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error('invalid percentile range')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')

    output_images = args.output / 'images'
    output_annotations = args.output / 'annfiles'
    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations.mkdir(parents=True, exist_ok=True)
    stride = args.tile_size - args.overlap
    manifest_rows = []
    group_rows = []

    image_paths = sorted(
        path for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        parser.error('no supported images found')

    for image_path in image_paths:
        annotation_path = args.annotations / f'{image_path.stem}.txt'
        if not annotation_path.is_file():
            parser.error(f'missing annotation file: {annotation_path}')
        records = parse_dota(annotation_path)
        with Image.open(image_path) as source:
            scene = to_uint8(np.asarray(source), args.intensity,
                             args.lower_percentile, args.upper_percentile)
        height, width = scene.shape
        for y in positions(height, args.tile_size, stride):
            for x in positions(width, args.tile_size, stride):
                chip_records = []
                for points, class_name, difficulty in records:
                    clipped = clipped_box(
                        points, x, y, args.tile_size, args.min_visible)
                    if clipped is not None:
                        chip_records.append((clipped, class_name, difficulty))
                if not chip_records and not keep_empty(
                        image_path.stem, x, y, args.empty_ratio, args.seed):
                    continue

                chip = np.zeros((args.tile_size, args.tile_size), dtype=np.uint8)
                visible = scene[y:min(y + args.tile_size, height),
                                x:min(x + args.tile_size, width)]
                chip[:visible.shape[0], :visible.shape[1]] = visible
                chip_stem = f'{image_path.stem}__{x}___{y}'
                Image.fromarray(chip).save(output_images / f'{chip_stem}.png')
                lines = []
                for points, class_name, difficulty in chip_records:
                    coords = ' '.join(
                        f'{float(value):.3f}' for point in points for value in point)
                    lines.append(f'{coords} {class_name} {difficulty}')
                (output_annotations / f'{chip_stem}.txt').write_text(
                    ('\n'.join(lines) + '\n') if lines else '')
                manifest_rows.append((chip_stem, image_path.stem,
                                      image_path.name, x, y, len(lines)))
                group_rows.append((chip_stem, image_path.stem))

    with (args.output / 'chips_manifest.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('stem', 'acquisition', 'source_image', 'x', 'y', 'objects'))
        writer.writerows(manifest_rows)
    with (args.output / 'groups.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('stem', 'acquisition'))
        writer.writerows(group_rows)

    print(f'created {len(manifest_rows)} chips under {args.output}')
    print(f'fold mapping -> {args.output / "groups.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
