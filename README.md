# SAR-GFM-Bench

A controlled benchmark for measuring how frozen vision and geospatial
foundation-model features transfer to high-resolution synthetic-aperture radar
(SAR) aircraft detection.

Every encoder is evaluated through the same trainable ViTDet-style adapter,
feature pyramid, and Oriented R-CNN detection head. The shared input contract is
a three-channel copy of the same 8-bit SAR amplitude image scaled to `[0, 1]`;
model-specific normalization and modality adaptation live in each backbone
wrapper.

## What is included

- Backbone wrappers for DINOv2, DINOv3, MAE ViT, SARATR-X, Clay,
  Copernicus-FM, CROMA, DOFA, Galileo, OlmoEarth, and Prithvi.
- A shared four-level ViTDet adapter emitting strides 8, 16, 32, and 64.
- Oriented R-CNN configs for four-fold cross-acquisition evaluation and an
  in-domain spatial split.
- Capella cross-sensor evaluation code.
- Training, evaluation, result collection, and dataset-conversion utilities.
- CPU shape tests for the shared adapter.

## Data and weights

**No imagery, annotations, model checkpoints, experiment logs, or generated
reports are included in this repository or its Git history.**

The Umbra benchmark data are not publicly redistributed. To run the benchmark,
provide a DOTA-style dataset through the paths expected by the GrokSAR fold
configs, or set `UMBRA_INDOMAIN_ROOT` for the in-domain split. Model weights
must be downloaded from their respective upstream projects and placed under
`weights/` as referenced by the selected config.

The expected DOTA-style split layout is:

```text
<split>/
|-- images/
`-- annfiles/
```

Each annotation is one quadrilateral per line followed by the class name and
difficulty flag.

## Environment

The benchmark was exercised with Python 3.10, PyTorch 2.0.1 + CUDA 11.8,
MMEngine 0.10.7, MMCV 2.0.1, MMDetection 3.0.0, and MMRotate 1.0.0rc1.
Install the matching PyTorch build first, then the OpenMMLab stack and remaining
Python packages:

```bash
python -m pip install openmim
mim install 'mmengine==0.10.7' 'mmcv==2.0.1' 'mmdet==3.0.0' 'mmrotate==1.0.0rc1'
python -m pip install -r requirements.txt
```

GrokSAR supplies the dataset and metric registrations used by the configs:

```bash
git clone https://github.com/GrokCV/GrokSAR.git ../GrokSAR
python -m pip install -e ../GrokSAR
```

Several wrappers use an upstream model repository. Clone only those needed for
your experiments, normally beside this repository:

- DINOv3: <https://github.com/rfaulk/DINO_Soars>
- Clay: <https://github.com/Clay-foundation/model>
- Galileo: <https://github.com/nasaharvest/galileo>
- OlmoEarth: <https://github.com/allenai/olmoearth_pretrain>

Override non-default locations with `GROKSAR_ROOT`, `DINOV3_ROOT`,
`SARATRX_ROOT`, `CLAY_REPO`, `GALILEO_REPO`, and `OLMO_REPO`.

## Quick checks

Run the adapter test without downloading model weights:

```bash
python tests/test_adapter.py
```

Train one frozen-backbone fold:

```bash
python tools/train.py   configs/umbra_cv/orcnn_dinov2l_frozen_fold0.py   --work-dir work_dirs/dinov2l_fold0
```

Run all four folds for a config tag:

```bash
tools/run_cv.sh dinov2l_frozen
```

Collect the best per-fold AP values from MMEngine logs:

```bash
python tools/collect_results.py --work-dir work_dirs
```

For Capella conversion, provide the source explicitly; generated chips remain
ignored by Git:

```bash
python data/capella_cvat_to_dota.py   --src /path/to/cvat-export   --out datasets/capella_test
```

## Repository layout

```text
adapters/    shared ViT-to-FPN feature adapter
backbones/   model wrappers and required vendored model components
configs/     shared detector and per-model experiment configs
data/        conversion utilities (no datasets)
evals/       evaluation adapters
tests/       lightweight checks
tools/       training, testing, and result collection
```

## Licensing

A project-level license has not yet been selected. Vendored third-party
components remain subject to their upstream licenses; the applicable texts and
notices are retained under `third_party_licenses/`.
