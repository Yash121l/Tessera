"""ML models (LightGBM primary/meta, PatchTST, TFT, Chronos, ensemble, registry)."""

from tessera.models.base import CVScores, Model, ModelCard, compute_deflated_sharpe
from tessera.models.ensemble import EnsembleModel
from tessera.models.lightgbm_model import MetaLightGBMModel, PrimaryLightGBMModel
from tessera.models.meta_model import MetaModel
from tessera.models.registry import ModelRegistry

# Sequence models: imported lazily to avoid hard torch/chronos dep at import time
# Use:  from tessera.models.patchtst import PatchTSTModel
#       from tessera.models.tft import TFTModel
#       from tessera.models.chronos_zeroshot import ChronosZeroShotModel

__all__ = [
    "CVScores",
    "EnsembleModel",
    "MetaLightGBMModel",
    "MetaModel",
    "Model",
    "ModelCard",
    "ModelRegistry",
    "PrimaryLightGBMModel",
    "compute_deflated_sharpe",
]
