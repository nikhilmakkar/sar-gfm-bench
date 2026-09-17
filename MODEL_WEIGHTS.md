# Model weights and source provenance

Pretrained weights are not redistributed here. Obtain them under their
upstream terms, put them at the expected path, and verify the SHA-256 value.
The hashes below identify the exact local artifacts used for the reported
experiments; a mismatched hash may represent a different upstream release or
conversion. The two DINOv3 `.pth` entries are the PyTorch artifacts converted
from the upstream SafeTensors downloads used by the wrapper.

| Model | Expected path | SHA-256 | Upstream |
|---|---|---|---|
| DINOv2-L | Torch Hub cache | managed by `facebookresearch/dinov2` | [DINOv2](https://github.com/facebookresearch/dinov2) |
| DINOv3-L web | `weights/dinov3/dinov3_vitl16_web.pth` | `7924a3d1587ccac892e5e7a37166f327e4db3866e7d752c2703ee2b39ffc9b05` | [DINO Soars integration](https://github.com/rfaulk/DINO_Soars) |
| DINOv3-L SAT-493M | `weights/dinov3/dinov3_vitl16_sat.pth` | `5d044c5f576fe5a0b425b7f2d35feb8171f889f500ac008355259406296551d7` | [DINO Soars integration](https://github.com/rfaulk/DINO_Soars) |
| ImageNet MAE ViT-L | `weights/mae_pretrain_vit_large.pth` | `d2ceff4892889d0c0f5418fa8bcb02e6a25a37fc3bdc0619c3cacd7d3d05c39f` | [MAE](https://github.com/facebookresearch/mae) |
| HiViT ImageNet MAE | `weights/saratrx/mae_hivit_base_1600ep.pth` | `7a3f1f45656788a69cc439fec24f4b9935f24404699ece108fe049d1e33def50` | [HiViT](https://github.com/zhangxiaosong18/hivit) |
| SARATR-X | `weights/saratrx/checkpoint-800.pth` | `dd9751b369172bd4532cf17cdd45d3750eb93fcb86bb05104e7084ba77dfbc8b` | [SARATR-X](https://github.com/waterdisappear/SARATR-X) |
| Clay v1.5 | `weights/clay/clay-v1.5.ckpt` | `21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0` | [Clay model](https://github.com/Clay-foundation/model) |
| Copernicus-FM | `weights/copernicusfm/CopernicusFM_ViT_large_varlang_e100.pth` | `e78a75c5b04a8980773de0a6f0efd4b237c3fb46c5180f397dd192de951a3ffc` | [Copernicus-FM](https://github.com/zhu-xlab/Copernicus-FM) |
| CROMA-large | `weights/croma/CROMA_large.pt` | `921e69ad4069201298bb5c860dc332f31e0cb01df6864f294a79fa8f921915a7` | [CROMA](https://github.com/antofuller/CROMA) |
| DOFA ViT-L | `weights/dofa/DOFA_ViT_large_e100.pth` | `ea9910bc026268ca36c30b76fd1169aeaca9ea51af5eaff840a20f150ea03d49` | [DOFA](https://github.com/zhu-xlab/DOFA) |
| Galileo-base | `weights/galileo_base/encoder.pt` | `935f4912f08483d052d69d8e96518f52ef6827fe718a435cb5536fd9bb7737a7` | [Galileo](https://github.com/nasaharvest/galileo) |
| Prithvi-EO-2.0-600M | `weights/prithvi/Prithvi_EO_V2_600M.pt` | `26a3f387ac6fc8e5eee8eb1986e6b9e99f36aa4d03aeb7fd5722feee5d5f88d0` | [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M) |
| OlmoEarth-v1-Large | `weights/olmoearth_large/weights.pth` | `1adb5026bd520c54bc415a1282386954927623bab81d01be2f5b6379cc039035` | [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) |

Additional required sidecars:

| File | SHA-256 |
|---|---|
| `weights/galileo_base/config.json` | `e405397099b02d0c1244e80e74d4b0584b6a35b06f53828ebacf3e75c0c4bcf6` |
| `weights/olmoearth_large/config.json` | `bd5f0fe3f571cf8beed64072d6c0029d6072223e6dce0b84b689c34d6638bbf1` |

Code revisions recorded from the checkouts used during the benchmark:

| Project | Revision |
|---|---|
| Clay | `f14e698f3c237cabf8d28dec669a362d66625381` |
| Copernicus-FM | `da95635d76a1feef3006eba474f6175325b4037b` |
| CROMA | `59505a6bcadbf36ba20767270154bf9f3067c5e7` |
| DOFA | `8346385695912606f74e00ef601b5c598a27df78` |
| Galileo | `0f0b5b95ac81acef4b74cf4686dc877202f4541b` |
| OlmoEarth | `08396f06972ded2dc4559ae8815fab01e690c01d` |

Some upstream checkpoints require accepting their license or model-card terms.
This repository's Apache-2.0 license does not override those terms.
