# Reference results

These are the authors' reported patch-level cross-acquisition results on the
internal annotation set. The metric is DOTA 11-point AP at IoU 0.5; each entry
is the best validation AP over 12 epochs. Because the annotations and original
fold membership cannot be released, these values are reference results rather
than independently reproducible leaderboard scores.

| Backbone | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---:|---:|---:|---:|---:|
| DINOv2-L frozen | 0.575 | 0.237 | 0.126 | 0.662 | **0.400** |
| SARATR-X HiViT-B finetuned | 0.494 | 0.285 | 0.123 | 0.676 | 0.394 |
| DINOv3-L/16 web frozen | 0.590 | 0.202 | 0.116 | 0.628 | 0.384 |
| ResNet-18 + Oriented R-CNN | 0.616 | 0.213 | 0.091 | 0.584 | 0.376 |
| DINOv3-L/16 SAT-493M frozen | 0.489 | 0.177 | 0.091 | 0.594 | 0.338 |
| ImageNet-MAE ViT-L frozen | 0.444 | 0.177 | 0.097 | 0.584 | 0.325 |
| ImageNet-MAE HiViT-B frozen | 0.414 | 0.167 | 0.113 | 0.558 | 0.313 |
| SARATR-X HiViT-B frozen | 0.368 | 0.149 | 0.073 | 0.527 | 0.279 |
| CROMA-large frozen | 0.340 | 0.126 | 0.091 | 0.524 | 0.270 |
| DOFA v1 ViT-L frozen | 0.330 | 0.169 | 0.091 | 0.482 | 0.268 |
| Prithvi-EO-2.0-600M frozen | 0.358 | 0.161 | 0.028 | 0.500 | 0.262 |
| OlmoEarth-v1-Large frozen | 0.391 | 0.127 | 0.008 | 0.485 | 0.253 |
| Galileo-base frozen | 0.383 | 0.118 | 0.018 | 0.465 | 0.246 |
| Clay v1.5 ViT-L frozen | 0.308 | 0.108 | 0.091 | 0.467 | 0.244 |
| Copernicus-FM ViT-L frozen | 0.315 | 0.123 | 0.091 | 0.369 | 0.224 |

The historical DenoDet baseline scored `0.313` mean AP. The central result is
that generic web-pretrained DINOv2-L was the strongest frozen representation,
while the largest remaining failure was generalization to fold 2.

For the analysis and conclusions, see the [companion Substack article](https://nikhilmakkar.substack.com/p/performance-of-geo-foundation-models). When citing these results, link to both the article and this repository.
