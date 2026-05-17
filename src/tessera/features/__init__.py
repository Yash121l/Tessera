"""Feature engineering pipeline."""

from tessera.features.base import Feature, FeaturePipeline
from tessera.features.cross_sectional import BetaToBTC, IdiosyncraticResidual, UniverseRank
from tessera.features.funding import FundingRate, FundingZScore, SpotPerpBasis
from tessera.features.microstructure import (
    VPIN,
    DepthWeightedSlippage,
    MicroPrice,
    OrderFlowImbalance,
    SpreadBps,
)
from tessera.features.regime import HMMRegime
from tessera.features.returns import LogReturn
from tessera.features.volatility import (
    Garch11,
    GarmanKlass,
    Parkinson,
    RealizedVol,
    VolOfVol,
)

__all__ = [
    "Feature",
    "FeaturePipeline",
    # Microstructure
    "OrderFlowImbalance",
    "MicroPrice",
    "VPIN",
    "SpreadBps",
    "DepthWeightedSlippage",
    # Volatility
    "RealizedVol",
    "Garch11",
    "Parkinson",
    "GarmanKlass",
    "VolOfVol",
    # Returns
    "LogReturn",
    # Cross-sectional
    "UniverseRank",
    "BetaToBTC",
    "IdiosyncraticResidual",
    # Funding
    "FundingRate",
    "FundingZScore",
    "SpotPerpBasis",
    # Regime
    "HMMRegime",
]
