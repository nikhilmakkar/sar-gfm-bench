#!/usr/bin/env python3
"""Convert a user-supplied CVAT image-task XML export to DOTA labels."""

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def rotated_box_points(
        xtl: float, ytl: float, xbr: float, ybr: float,
        rotation_degrees: float) -> list[tuple[float, float]]:
    cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
    half_w, half_h = (xbr - xtl) / 2, (ybr - ytl) / 2
    angle = math.radians(rotation_degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = []
    for dx, dy in ((-half_w, -half_h), (half_w, -half_h),
                   (half_w, half_h), (-half_w, half_h)):
        points.append((cx + dx * cos_a - dy * sin_a,
                       cy + dx * sin_a + dy * cos_a))
    return points


def shape_points(shape: ET.Element) -> list[tuple[float, float]]:
    if shape.tag == 'polygon':
        points = []
        for pair in shape.attrib['points'].split(';'):
            x, y = pair.split(',')
            points.append((float(x), float(y)))
        if len(points) != 4:
            raise ValueError(
                f'expected a four-point polygon, found {len(points)} points')
        return points
    if shape.tag == 'box':
        return rotated_box_points(
            float(shape.attrib['xtl']), float(shape.attrib['ytl']),
            float(shape.attrib['xbr']), float(shape.attrib['ybr']),
            float(shape.attrib.get('rotation', 0)))
    raise ValueError(f'unsupported CVAT shape: {shape.tag}')


def is_difficult(shape: ET.Element, attribute_name: str) -> bool:
    if truthy(shape.attrib.get('occluded')):
        return True
    for attribute in shape.findall('attribute'):
        if attribute.attrib.get('name') == attribute_name:
            return truthy(attribute.text)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Convert your CVAT image-task XML export to DOTA text.')
    parser.add_argument('cvat_xml', type=Path,
                        help='your own CVAT XML export (filename is arbitrary)')
    parser.add_argument('output', type=Path,
                        help='new directory for full-scene DOTA .txt files')
    parser.add_argument(
        '--include-label', action='append', default=[],
        help='CVAT label to include; repeatable. By default all labels are included')
    parser.add_argument('--class-name', default='aircraft',
                        help='single output class name (default: aircraft)')
    parser.add_argument('--difficulty-attribute', default='difficult')
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)

    root = ET.parse(args.cvat_xml).getroot()
    included = set(args.include_label)
    seen_stems: set[str] = set()
    manifest_rows = []
    total_objects = 0

    for image in root.findall('.//image'):
        source_name = image.attrib['name']
        stem = Path(source_name).stem
        if stem in seen_stems:
            parser.error(
                f'duplicate image stem "{stem}"; rename images uniquely before export')
        seen_stems.add(stem)
        width, height = int(image.attrib['width']), int(image.attrib['height'])
        lines = []
        for shape in image:
            if shape.tag not in {'polygon', 'box'}:
                continue
            if truthy(shape.attrib.get('outside')):
                continue
            label = shape.attrib.get('label', '')
            if included and label not in included:
                continue
            try:
                points = shape_points(shape)
            except ValueError as exc:
                parser.error(f'{source_name}: {exc}')
            coords = ' '.join(f'{value:.3f}' for point in points for value in point)
            difficulty = int(is_difficult(shape, args.difficulty_attribute))
            lines.append(f'{coords} {args.class_name} {difficulty}')
        (args.output / f'{stem}.txt').write_text(
            ('\n'.join(lines) + '\n') if lines else '')
        manifest_rows.append((stem, source_name, width, height, len(lines)))
        total_objects += len(lines)

    with (args.output / 'scene_manifest.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('stem', 'source_name', 'width', 'height', 'objects'))
        writer.writerows(manifest_rows)

    print(f'converted {len(manifest_rows)} scenes and {total_objects} objects')
    print(f'full-scene labels -> {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
