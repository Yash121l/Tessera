"""File-based model registry.

Stores trained models under ``models/{model_name}/{git_sha}_{timestamp}/``
with a ``current`` symlink pointing at the latest promoted version.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import structlog

from tessera.config import TesseraSettings
from tessera.models.base import Model, ModelCard, get_git_commit

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Manages versioned model storage and promotion."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            root = TesseraSettings().models_root
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Model, model_name: str) -> Path:
        """Persist *model* and return the version directory."""
        git_sha = get_git_commit()[:8]
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        version_dir = self.root / model_name / f"{git_sha}_{ts}"
        model.save(version_dir)
        logger.info("model_saved", model_name=model_name, path=str(version_dir))
        return version_dir

    def promote(self, model_path: Path, min_sharpe: float = 0.0) -> Path:
        """Point the ``current`` symlink at *model_path* after validation.

        Raises ``ValueError`` if the model card's mean CV Sharpe is below
        *min_sharpe*.
        """
        card_path = model_path / "model_card.json"
        if not card_path.exists():
            msg = f"No model_card.json found in {model_path}"
            raise FileNotFoundError(msg)

        card = ModelCard.model_validate_json(card_path.read_text())

        if card.cv_scores is not None and card.cv_scores.mean_sharpe < min_sharpe:
            msg = (
                f"CV Sharpe {card.cv_scores.mean_sharpe:.4f} "
                f"below promotion threshold {min_sharpe:.4f}"
            )
            raise ValueError(msg)

        model_name = model_path.parent.name
        current_link = self.root / model_name / "current"
        if current_link.is_symlink() or current_link.exists():
            current_link.unlink()
        current_link.symlink_to(model_path.resolve())
        logger.info("model_promoted", model_name=model_name, path=str(model_path))
        return current_link

    def load_current(self, model_name: str, model_cls: type[Model]) -> Model:
        """Load the currently promoted model."""
        current = self.root / model_name / "current"
        if not current.exists():
            msg = f"No promoted model for '{model_name}'"
            raise FileNotFoundError(msg)
        return model_cls.load(current)

    def list_versions(self, model_name: str) -> list[Path]:
        """Return all version directories for *model_name*, sorted by name."""
        model_dir = self.root / model_name
        if not model_dir.exists():
            return []
        return sorted(
            [p for p in model_dir.iterdir() if p.is_dir() and p.name != "current"],
            key=lambda p: p.name,
        )
