"""Shape test for ViTDetAdapter — runs on CPU, no downloads."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from adapters.vitdet_adapter import ViTDetAdapter, FPN_STRIDES


def test_shapes(patch, H, W, embed_dim=768):
    grid_h, grid_w = H // patch if H % patch == 0 else round(H / patch), \
        W // patch if W % patch == 0 else round(W / patch)
    # simulate wrapper: encoder ran at nearest patch multiple
    gh = max(1, round(H / patch))
    gw = max(1, round(W / patch))
    feats = [torch.randn(2, embed_dim, gh, gw) for _ in range(4)]
    adapter = ViTDetAdapter(embed_dim, out_channels=256)
    outs = adapter(feats, (H, W))
    assert len(outs) == 4
    for o, s in zip(outs, FPN_STRIDES):
        expect = (2, 256, H // s, W // s)
        assert tuple(o.shape) == expect, f'{tuple(o.shape)} != {expect}'
    print(f'patch={patch:>2} input={H}x{W}: '
          + ', '.join(str(tuple(o.shape[-2:])) for o in outs) + '  OK')


if __name__ == '__main__':
    test_shapes(16, 512, 512)
    test_shapes(14, 512, 512)   # fractional grid case
    test_shapes(14, 518, 518)
    test_shapes(8, 256, 256)
    # gradient flows through lateral convs
    adapter = ViTDetAdapter(384, 256)
    feats = [torch.randn(1, 384, 37, 37) for _ in range(4)]
    outs = adapter(feats, (512, 512))
    sum(o.sum() for o in outs).backward()
    g = adapter.lateral_convs[0][0].weight.grad
    assert g is not None and g.abs().sum() > 0
    print('adapter gradient flow OK')
