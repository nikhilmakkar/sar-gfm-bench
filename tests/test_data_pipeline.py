#!/usr/bin/env python3
"""End-to-end smoke test for the public-data preparation path."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: object) -> None:
    subprocess.run(
        [sys.executable, *(str(argument) for argument in arguments)],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        scenes = root / 'scenes'
        scenes.mkdir()
        image_entries = []

        for index in range(4):
            name = f'scene_{index}.tif'
            values = np.arange(96 * 96, dtype=np.uint16).reshape(96, 96)
            Image.fromarray(values + index).save(scenes / name)
            image_entries.append(
                f'<image id="{index}" name="{name}" width="96" height="96">'
                '<polygon label="plane" points="20,20;44,20;44,36;20,36"/>'
                '</image>'
            )

        cvat_export = root / 'any_user_chosen_name.xml'
        cvat_export.write_text(
            '<annotations>' + ''.join(image_entries) + '</annotations>'
        )
        full_labels = root / 'full_labels'
        chips = root / 'chips'
        folds = root / 'folds'

        run('tools/cvat_to_dota.py', cvat_export, full_labels,
            '--include-label', 'plane')
        run('tools/tile_dota_scenes.py', scenes, full_labels, chips,
            '--tile-size', 64, '--overlap', 16, '--empty-ratio', 1)
        run('tools/validate_dota_dataset.py', chips)
        run('tools/make_acquisition_folds.py', chips, chips / 'groups.csv',
            folds, '--folds', 4, '--copy')

        assert len(list((chips / 'images').glob('*.png'))) == 16
        assert len(list(folds.glob('fold_*'))) == 4

    print('public data pipeline smoke test passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
