"""PatchTST multivariate time-series classifier (Nie et al., ICLR 2023).

Adapted from the forecasting formulation to triple-barrier classification.
Each row of the input DataFrame is treated as one time step; the model builds
sliding windows of length ``lookback`` internally so the API stays compatible
with the tabular Model ABC.

Architecture (≤ 5M params for any reasonable feature set):
  lookback=60, patch_len=8  → 8 patches per feature (pad seq to 64)
  d_model=128, n_heads=4, n_layers=3, ffn_dim=256, dropout=0.1

Label mapping: {-1 → 0, 0 → 1, +1 → 2} for CrossEntropyLoss; reversed at output.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tessera.config import TesseraSettings
from tessera.models.base import CVScores, Model, ModelCard, compute_deflated_sharpe, get_git_commit

logger = structlog.get_logger(__name__)

# Architecture constants — keep ≤5M params
_LOOKBACK = 60
_PATCH_LEN = 8
_D_MODEL = 128
_N_HEADS = 4
_N_LAYERS = 3
_FFN_DIM = 256
_DROPOUT = 0.1
_N_CLASSES = 3

# Triple-barrier label ↔ class-index mapping
_LABEL_TO_IDX: dict[int, int] = {-1: 0, 0: 1, 1: 2}
_IDX_TO_LABEL: dict[int, int] = {0: -1, 1: 0, 2: 1}


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class _PatchTSTNet(nn.Module):
    """Pure-PyTorch PatchTST encoder + classification head."""

    def __init__(
        self,
        n_features: int,
        lookback: int = _LOOKBACK,
        patch_len: int = _PATCH_LEN,
        d_model: int = _D_MODEL,
        n_heads: int = _N_HEADS,
        n_layers: int = _N_LAYERS,
        ffn_dim: int = _FFN_DIM,
        dropout: float = _DROPOUT,
        n_classes: int = _N_CLASSES,
    ) -> None:
        super().__init__()
        # Pad lookback to nearest multiple of patch_len
        self.padded_len = math.ceil(lookback / patch_len) * patch_len
        self.patch_len = patch_len
        n_patches = self.padded_len // patch_len
        self.n_tokens = n_features * n_patches  # one token per (feature, patch) pair

        # Shared linear projection: raw patch → d_model
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Learnable positional embedding over the flat token sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN stabilises deep transformers
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, lookback, n_features]
        batch, t, _ = x.shape

        # Zero-pad sequence dimension to padded_len
        pad = self.padded_len - t
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))

        # Create patches: [batch, n_patches, n_features, patch_len]
        x = x.unfold(1, self.patch_len, self.patch_len)

        # Rearrange to [batch, n_features, n_patches, patch_len] → project → d_model
        x = x.permute(0, 2, 1, 3)
        x = self.patch_proj(x)

        # Flatten feature × patch dims → [batch, n_tokens, d_model]
        x = x.reshape(batch, self.n_tokens, -1)
        x = self.dropout(x + self.pos_embed)

        # Transformer + post-norm
        x = self.encoder(x)
        x = self.norm(x)

        # Global mean pooling → [B, d_model]
        x = x.mean(dim=1)

        return self.head(x)  # [B, n_classes]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    """Set all relevant random seeds for CPU-deterministic training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Best-effort: CUDA ops may still be non-deterministic on some GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_sequences(arr: np.ndarray, lookback: int) -> np.ndarray:
    """Sliding-window view: [n_bars, n_features] → [n_bars-lookback+1, lookback, n_features]."""
    n_bars = arr.shape[0]
    n = n_bars - lookback + 1
    if n <= 0:
        msg = f"Need ≥ {lookback} bars, got {n_bars}"
        raise ValueError(msg)
    n_feats = arr.shape[1]
    shape = (n, lookback, n_feats)
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides).copy()


# ---------------------------------------------------------------------------
# PatchTST Model (Model ABC)
# ---------------------------------------------------------------------------


class PatchTSTModel(Model):
    """PatchTST classifier for triple-barrier labels.

    Stores the last ``lookback`` rows of training data so that ``predict()``
    can be called on a contiguous validation DataFrame without losing the first
    window's context.
    """

    def __init__(
        self,
        lookback: int = _LOOKBACK,
        patch_len: int = _PATCH_LEN,
        d_model: int = _D_MODEL,
        n_heads: int = _N_HEADS,
        n_layers: int = _N_LAYERS,
        ffn_dim: int = _FFN_DIM,
        dropout: float = _DROPOUT,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 256,
        max_epochs: int = 50,
        patience: int = 10,
        seed: int | None = None,
    ) -> None:
        settings = TesseraSettings()
        self._seed = seed if seed is not None else settings.random_seed
        self._lookback = lookback
        self._patch_len = patch_len
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._ffn_dim = ffn_dim
        self._dropout = dropout
        self._lr = lr
        self._weight_decay = weight_decay
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience

        self._net: _PatchTSTNet | None = None
        self._context: np.ndarray | None = None  # last lookback rows of training data
        self._feature_names: list[str] = []
        self._cv_scores: CVScores | None = None
        self._training_date: str = ""
        self._data_version: str = ""

    # ------------------------------------------------------------------
    # Internal training loop
    # ------------------------------------------------------------------

    def _build_net(self, n_features: int) -> _PatchTSTNet:
        _set_seed(self._seed)
        net = _PatchTSTNet(
            n_features=n_features,
            lookback=self._lookback,
            patch_len=self._patch_len,
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_layers=self._n_layers,
            ffn_dim=self._ffn_dim,
            dropout=self._dropout,
        )
        logger.info(
            "patchtst_params",
            n_params=net.count_parameters(),
            n_features=n_features,
        )
        return net

    def _train_loop(
        self,
        seqs_train: np.ndarray,
        labels_train: np.ndarray,
        seqs_val: np.ndarray,
        labels_val: np.ndarray,
        class_weights: torch.Tensor,
    ) -> None:
        assert self._net is not None

        device = torch.device("cpu")
        net = self._net.to(device)

        x_tr = torch.tensor(seqs_train, dtype=torch.float32)
        y_tr = torch.tensor(labels_train, dtype=torch.long)
        x_va = torch.tensor(seqs_val, dtype=torch.float32)
        y_va = torch.tensor(labels_val, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(self._seed),
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self._max_epochs,
            eta_min=1e-6,
        )

        best_val_loss = float("inf")
        best_state: dict[str, Any] = {}
        no_improve = 0

        for epoch in range(self._max_epochs):
            net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            net.eval()
            with torch.no_grad():
                val_loss = float(criterion(net(x_va), y_va).item())

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0:
                logger.debug(
                    "patchtst_epoch",
                    epoch=epoch,
                    val_loss=round(val_loss, 4),
                    no_improve=no_improve,
                )

            if no_improve >= self._patience:
                logger.info("patchtst_early_stop", epoch=epoch, val_loss=best_val_loss)
                break

        if best_state:
            net.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_sequences_for_fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = X.values.astype(np.float32)
        seqs = _make_sequences(arr, self._lookback)  # [N-L+1, L, F]
        # Align labels: first valid label starts at index lookback-1
        labels_raw = y.values[self._lookback - 1 :]
        labels = np.array([_LABEL_TO_IDX[int(v)] for v in labels_raw], dtype=np.int64)
        return seqs, labels

    def _prepare_input(self, X: pd.DataFrame) -> torch.Tensor:  # noqa: N803
        """Build sequences from X, prepending stored training context."""
        arr = X.values.astype(np.float32)

        if self._context is not None and len(self._context) >= self._lookback:
            prefix = self._context[-self._lookback :]
            full = np.concatenate([prefix, arr], axis=0)
        else:
            # Zero-pad prefix so every row of X gets a lookback window
            pad = np.zeros((self._lookback, arr.shape[1]), dtype=np.float32)
            full = np.concatenate([pad, arr], axis=0)

        n = len(arr)
        seqs = np.stack([full[i : i + self._lookback] for i in range(n)], axis=0)
        return torch.tensor(seqs, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,  # type: ignore[type-arg]
        sample_weight: np.ndarray | None = None,
        X_val: pd.DataFrame | None = None,  # noqa: N803
        y_val: pd.Series | None = None,  # type: ignore[type-arg]
    ) -> Self:
        _set_seed(self._seed)
        self._feature_names = list(X.columns)
        self._training_date = datetime.now(UTC).isoformat()
        n_features = len(self._feature_names)

        self._net = self._build_net(n_features)

        seqs_train, labels_train = self._prepare_sequences_for_fit(X, y)

        # Validation fold: use provided X_val or last 15% of training seqs
        if X_val is not None and y_val is not None:
            seqs_val, labels_val = self._prepare_sequences_for_fit(X_val, y_val)
        else:
            split = max(1, int(0.85 * len(seqs_train)))
            seqs_val = seqs_train[split:]
            labels_val = labels_train[split:]
            seqs_train = seqs_train[:split]
            labels_train = labels_train[:split]

        # Class weights to handle label imbalance (0-class is often dominant)
        counts = np.bincount(labels_train, minlength=_N_CLASSES).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        weights = torch.tensor(1.0 / counts, dtype=torch.float32)
        weights = weights / weights.sum()

        self._train_loop(seqs_train, labels_train, seqs_val, labels_val, weights)

        # Store context for sequential predict calls
        self._context = X.values.astype(np.float32)[-self._lookback :]

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        assert self._net is not None, "Call fit() first"
        self._net.eval()
        x_t = self._prepare_input(X)
        with torch.no_grad():
            logits = self._net(x_t)
        idx = logits.argmax(dim=-1).numpy()
        return np.array([_IDX_TO_LABEL[int(i)] for i in idx])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        assert self._net is not None, "Call fit() first"
        self._net.eval()
        x_t = self._prepare_input(X)
        with torch.no_grad():
            logits = self._net(x_t)
        return torch.softmax(logits, dim=-1).numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        assert self._net is not None
        torch.save(self._net.state_dict(), path / "weights.pt")
        if self._context is not None:
            np.save(str(path / "context.npy"), self._context)
        arch = {
            "n_features": len(self._feature_names),
            "lookback": self._lookback,
            "patch_len": self._patch_len,
            "d_model": self._d_model,
            "n_heads": self._n_heads,
            "n_layers": self._n_layers,
            "ffn_dim": self._ffn_dim,
            "dropout": self._dropout,
        }
        (path / "arch.json").write_text(json.dumps(arch, indent=2))
        card = self.get_model_card()
        (path / "model_card.json").write_text(card.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        arch = json.loads((path / "arch.json").read_text())
        card_data = json.loads((path / "model_card.json").read_text())
        card = ModelCard.model_validate(card_data)

        instance = cls.__new__(cls)
        instance._lookback = arch["lookback"]
        instance._patch_len = arch["patch_len"]
        instance._d_model = arch["d_model"]
        instance._n_heads = arch["n_heads"]
        instance._n_layers = arch["n_layers"]
        instance._ffn_dim = arch["ffn_dim"]
        instance._dropout = arch["dropout"]
        instance._seed = 42
        instance._lr = 1e-3
        instance._weight_decay = 1e-2
        instance._batch_size = 256
        instance._max_epochs = 50
        instance._patience = 10
        instance._feature_names = card.features
        instance._cv_scores = card.cv_scores
        instance._training_date = card.training_date
        instance._data_version = card.data_version

        net = _PatchTSTNet(**arch)
        net.load_state_dict(torch.load(path / "weights.pt", map_location="cpu"))
        net.eval()
        instance._net = net

        ctx_path = path / "context.npy"
        instance._context = np.load(str(ctx_path)) if ctx_path.exists() else None

        return instance

    def get_model_card(self) -> ModelCard:
        n_params = self._net.count_parameters() if self._net else 0
        return ModelCard(
            model_name="patchtst",
            model_type="patchtst",
            training_date=self._training_date,
            git_commit=get_git_commit(),
            data_version=self._data_version,
            features=self._feature_names,
            hyperparameters={
                "lookback": self._lookback,
                "patch_len": self._patch_len,
                "d_model": self._d_model,
                "n_heads": self._n_heads,
                "n_layers": self._n_layers,
                "n_params": n_params,
                "lr": self._lr,
                "max_epochs": self._max_epochs,
            },
            cv_scores=self._cv_scores,
        )

    # ------------------------------------------------------------------
    # CV Sharpe (mirrors LightGBM's _compute_cv_sharpe pattern)
    # ------------------------------------------------------------------

    def cv_sharpe(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,  # type: ignore[type-arg]
        t1: pd.Series,  # type: ignore[type-arg]
        forward_returns: pd.Series | None = None,  # type: ignore[type-arg]
        n_splits: int = 5,
        pct_embargo: float = 0.01,
        n_trials: int = 1,
    ) -> CVScores:
        """Compute purged K-fold CV Sharpe ratio."""
        from tessera.cv.purged_kfold import PurgedKFold

        cv = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, pct_embargo=pct_embargo)
        fold_sharpes: list[float] = []
        n_obs = 0

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
            logger.info("patchtst_cv_fold", fold=fold_idx, n_train=len(train_idx))
            x_tr, x_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            self.fit(x_tr, y_tr, X_val=x_va, y_val=y_va)
            preds = self.predict(x_va)
            n_obs += len(val_idx)

            if forward_returns is not None:
                ret = forward_returns.iloc[val_idx].values
                strat = preds.astype(float) * ret
                std = float(strat.std())
                sharpe = float(strat.mean() / std) if std > 1e-12 else 0.0
            else:
                sharpe = float((preds == y_va.values).mean())

            fold_sharpes.append(sharpe)

        arr = np.array(fold_sharpes)
        self._cv_scores = CVScores(
            mean_sharpe=float(arr.mean()),
            std_sharpe=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            deflated_sharpe=compute_deflated_sharpe(arr, n_obs, n_trials),
            n_trials=n_trials,
        )
        return self._cv_scores
