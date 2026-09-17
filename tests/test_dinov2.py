"""GPU smoke test for DINOv2Backbone. Downloads weights on first run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from backbones.group_a.dinov2_backbone import DINOv2Backbone

variant = sys.argv[1] if len(sys.argv) > 1 else 'vits14'

model = DINOv2Backbone(variant=variant, frozen=True).cuda()
x = torch.rand(2, 3, 512, 512).cuda()

with torch.autocast('cuda', dtype=torch.bfloat16):
    outs = model(x)

for o, s in zip(outs, (8, 16, 32, 64)):
    print(f'stride {s:>2}: {tuple(o.shape)} dtype={o.dtype}')
    assert tuple(o.shape[-2:]) == (512 // s, 512 // s)
    assert o.shape[1] == 256

# frozen encoder must produce no grads; adapter must
loss = sum(o.float().sum() for o in outs)
loss.backward()
enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
ada_grad = model.adapter.lateral_convs[0][0].weight.grad
assert not enc_grads, 'encoder got gradients while frozen!'
assert ada_grad is not None and ada_grad.abs().sum() > 0
print(f'freeze/grad contract OK. peak mem '
      f'{torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
