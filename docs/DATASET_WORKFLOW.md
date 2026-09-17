# Build a runnable dataset from Umbra Open Data

The imagery and the labels have different release conditions:

- Umbra Open Data imagery is public under CC BY 4.0. Download it from the
  [Umbra Open Data Program](https://umbra.space/open-data/) or the
  [AWS Open Data Registry](https://registry.opendata.aws/umbra-open-data/).
- The annotations used for the reported results were created internally and
  cannot be redistributed.

You therefore cannot reconstruct the authors' exact labels or scores. You can
use the same benchmark implementation with your own annotations.

## 1. Select and download imagery

Use Umbra's catalog browser or [`umbra-py`](https://umbra-py.space/quickstart/)
to search by location, date, collection geometry, and product. Prefer a
geocoded detected product such as GEC if the goal is visual object detection.
Keep each scene's STAC JSON beside the raster so acquisition provenance is not
lost. Put the chosen public asset URLs and filenames in a copy of
`examples/umbra_download_manifest.csv`, then download and optionally checksum
them:

```bash
python tools/download_umbra.py examples/umbra_download_manifest.csv data/umbra
```

For a cross-acquisition study, select multiple independent collects. A larger
number of scenes from one collect does not replace diversity across dates,
satellites, incidence angles, resolutions, and look directions.

## 2. Annotate oriented aircraft boxes

Annotate one tight four-corner polygon around each aircraft. Use a single
class, `aircraft`, if you want to match this repository's detector head.

Recommended rules:

- Include every aircraft class consistently; do not filter by aircraft name.
- Keep the box tight to the visible aircraft extent, excluding shadow.
- Mark truncated or genuinely ambiguous objects as difficult.
- Review empty scenes and dense parking areas explicitly.
- Record the source scene/acquisition ID for every exported chip.

Export a CVAT **image-task XML** file. This is your own export and its filename
is arbitrary; `annotations.xml` is only an example name, not a file supplied or
licensed by this repository. Convert the export to full-scene DOTA labels:

```bash
python tools/cvat_to_dota.py \
  /path/to/my_cvat_export.xml build/full_scene_labels
```

The runtime consumes DOTA text files. Each image has a same-stem `.txt` file:

```text
x1 y1 x2 y2 x3 y3 x4 y4 aircraft 0
```

Coordinates are pixel coordinates. The final field is the DOTA difficulty
flag (`0` or `1`). Empty images need an empty annotation file.

## 3. Create 512 x 512 chips

Tile the downloaded scenes and transform their quadrilaterals automatically:

```bash
python tools/tile_dota_scenes.py \
  data/umbra build/full_scene_labels build/chips \
  --tile-size 512 --overlap 128 --min-visible 0.7
```

The command writes:

```text
my_chips/
|-- images/
|   |-- scene_a__0___0.png
|   `-- scene_b__0___0.png
`-- annfiles/
    |-- scene_a__0___0.txt
    `-- scene_b__0___0.txt
```

The reported experiments used 512-pixel chips. When tiling a large scene,
use overlap so boundary aircraft are not systematically discarded. Decide and
document a minimum-visible-area rule for clipped objects. Do not randomly
split chips from the same scene or acquisition across training and validation.

The tiler uses deterministic log-percentile intensity scaling by default and
writes `chips_manifest.csv` plus `groups.csv`. Inspect representative chips,
then validate the result before training:

```bash
python tools/validate_dota_dataset.py build/chips
```

## 4. Build acquisition-grouped folds

The tiler already creates a `groups.csv` mapping every chip to its source
scene. If several image files came from the same acquisition, edit the
`acquisition` column so those files share one acquisition ID. Then build four
folds, balanced approximately by labelled object count:

```bash
python tools/make_acquisition_folds.py \
  build/chips build/chips/groups.csv datasets/Umbra/umbra_cv
python tools/validate_dota_dataset.py datasets/Umbra/umbra_cv/fold_0
```

The tool keeps every acquisition wholly in train or validation. Use
`--always-train ACQUISITION` only when that asymmetry is intentional and report
it with the results. Pass `--copy` on systems where symlinks are inconvenient.

## 5. Run the benchmark

Point the portable configs at the generated fold root:

```bash
export UMBRA_CV_ROOT=/absolute/path/to/datasets/Umbra/umbra_cv
export GROKSAR_ROOT=/absolute/path/to/GrokSAR
python tools/train.py \
  configs/umbra_cv/orcnn_dinov2l_frozen_fold0.py \
  --work-dir work_dirs/dinov2l_fold0
```

Run all four folds with `tools/run_cv.sh dinov2l_frozen`, then aggregate the
logs with `python tools/collect_results.py --work-dir work_dirs`.

## What is and is not comparable

Results on newly created labels are valid uses of the framework, but are not
direct reproductions of the private-label reference table. Publish the scene
selection, annotation rules, chip generation parameters, group CSV, folds,
seeds, and model-weight versions with any new result.

Umbra imagery must retain the attribution required by CC BY 4.0. User-created
annotations remain governed by the license selected by their creator.
