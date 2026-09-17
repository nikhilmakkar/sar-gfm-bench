"""Importing this package registers all backbone wrappers with mmdet MODELS.

Group B/C imports are added as their wrappers are implemented; a missing
optional dependency for one model must not break the others, hence the
per-module try/except.
"""

from backbones.base_gfm import BaseGFMBackbone  # noqa: F401

_REGISTERED = []
_FAILED = {}


def _try(modpath):
    import importlib
    try:
        importlib.import_module(modpath)
        _REGISTERED.append(modpath)
    except Exception as e:  # noqa: BLE001
        _FAILED[modpath] = repr(e)


_try('backbones.group_a.dinov2_backbone')
_try('backbones.group_a.dinov3_backbone')
_try('backbones.group_a.saratrx_backbone')
_try('backbones.group_a.mae_vit_backbone')
_try('backbones.group_b.copernicusfm_backbone')
_try('backbones.group_b.dofa_backbone')
_try('backbones.group_b.clay_backbone')
_try('backbones.group_b.galileo_backbone')
_try('backbones.group_b.prithvi_backbone')
_try('backbones.group_b.croma_backbone')
_try('backbones.group_b.olmoearth_backbone')

if _FAILED:
    import warnings
    warnings.warn(f'some backbones failed to register: {_FAILED}')
