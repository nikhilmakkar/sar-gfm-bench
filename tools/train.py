"""Training entry point.

Puts both repo roots on sys.path (sar-gfm-bench for `backbones`/`adapters`,
GrokSAR for the `groksar` dataset/metric package) before mmengine parses the
config, then defers to the standard mmrotate training flow.

Usage:
    python tools/train.py configs/umbra_cv/orcnn_dinov2l_frozen_fold0.py \
        --work-dir work_dirs/dinov2l_fold0
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
    parser.add_argument('--work-dir', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--cfg-options', nargs='+', default=None,
        help='override config entries, key=value')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get('work_dir'):
        stem = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join('work_dirs', stem)
    if args.resume:
        cfg.resume = True
    if args.cfg_options:
        opts = dict(kv.split('=', 1) for kv in args.cfg_options)
        cfg.merge_from_dict(opts)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
