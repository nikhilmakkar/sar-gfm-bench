"""Evaluation entry point (mirrors tools/train.py path setup).

Usage:
    python tools/test.py CONFIG --checkpoint CHECKPOINT [--work-dir DIR]
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROKSAR_ROOT = os.environ.get(
    'GROKSAR_ROOT', os.path.join(os.path.dirname(REPO_ROOT), 'GrokSAR'))
for p in (REPO_ROOT, GROKSAR_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--work-dir', default=None)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get('work_dir'):
        stem = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join('work_dirs', stem)

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
