"""Pre-flight memory/correctness probe for CROMA, run before the real sweep.

CROMA has no flash/SDPA attention path (dense einsum + ALiBi bias), so its
memory scaling couldn't be safely profiled at the real training batch size
while the GPU was busy with Galileo/Prithvi (batch1=4.47GB, batch2=7.27GB;
batch4 OOM'd only because ~10GB was free at the time, not necessarily at
batch4 itself on a clear GPU). Per the sharpened smoke-test rule (training
batch size, >=2 steady-state iterations), this runs that real check right
before the sweep launches, so a bad batch size fails fast with one clear
message instead of 5 configs crashing in 40s under a false COMPLETE marker.
"""
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GROKSAR_ROOT = Path(os.environ.get(
    'GROKSAR_ROOT', REPO_ROOT.parent / 'GrokSAR'))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GROKSAR_ROOT))

import torch
import backbones  # noqa: F401  (registers CromaBackbone)
from backbones.group_b.croma_backbone import CromaBackbone

BATCH = 6

m = CromaBackbone(checkpoint='weights/croma/CROMA_large.pt', frozen=True,
                  bf16=True, image_resolution=512).cuda()
m.train()
assert not m.encoder.training, 'encoder not pinned to eval — freeze broken'

x = torch.rand(BATCH, 3, 512, 512, device='cuda')
outs = m(x)
sum(o.float().sum() for o in outs).backward()
enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
              for p in m.encoder.parameters())
adp_grad = all(p.grad is not None for p in m.adapter.parameters())
assert not enc_grad, 'encoder received gradient — freeze broken'
assert adp_grad, 'adapter did not receive gradient'
m.zero_grad()

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
times = []
for _ in range(3):
    t0 = time.time()
    outs = m(x)
    sum(o.float().sum() for o in outs).backward()
    torch.cuda.synchronize()
    times.append(time.time() - t0)
    m.zero_grad()

peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f'PREFLIGHT OK: batch={BATCH} peak_mem_gb={peak_gb:.2f} '
      f'iter_times={[round(t, 2) for t in times]}')
if peak_gb > 40:
    print(f'PREFLIGHT WARN: peak {peak_gb:.2f}GB is close to the 46GB card '
          f'limit; leaves little headroom for the trainable FPN/head.')
