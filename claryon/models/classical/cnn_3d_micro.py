"""Micro 3D CNN — minimal capacity classical comparator for quantum models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "cnn_3d_micro")
class CNN3DMicroModel(ModelBuilder):
    """Deliberately minimal 3D CNN (5–10K parameters).

    Provides a classical comparator closer in parameter count to the
    291-parameter qCNN. Architecture:
    Conv3d(1→4, 3³) → BN → ReLU → MaxPool(2)
    Conv3d(4→8, 3³) → BN → ReLU → MaxPool(2)
    Conv3d(8→16, 3³) → BN → ReLU → AdaptiveAvgPool(1)
    Flatten → Linear(16 → n_classes)

    Expects input X of shape (N, C, D, H, W).
    """

    def __init__(
        self,
        n_classes: int = 2,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 4,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self._n_classes = n_classes
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._seed = seed
        self._model: Any = None
        self._task_type = TaskType.BINARY

    @property
    def name(self) -> str:
        return "cnn_3d_micro"

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE_3D

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS)

    def _build_net(self, in_channels: int) -> Any:
        """Build the micro 3D CNN."""
        import torch.nn as nn

        net = nn.Sequential(
            # Block 1: 1 → 4
            nn.Conv3d(in_channels, 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # Block 2: 4 → 8
            nn.Conv3d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # Block 3: 8 → 16
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            # Classifier
            nn.Flatten(),
            nn.Linear(16, self._n_classes),
        )

        n_params = sum(p.numel() for p in net.parameters())
        logger.info("cnn_3d_micro: %d trainable parameters", n_params)
        return net

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Train micro 3D CNN.

        Args:
            X: Volume batch, shape (N, C, D, H, W).
            y: Integer labels.
            task_type: BINARY or MULTICLASS.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self._seed)
        self._task_type = task_type

        n_classes = len(np.unique(y))
        self._n_classes = n_classes

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        self._model = self._build_net(X.shape[1])
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        self._model.train()
        for epoch in range(self._epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self._model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.debug("cnn_3d_micro epoch %d/%d loss=%.4f", epoch + 1, self._epochs, total_loss / len(loader))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        import torch

        if self._model is None:
            raise RuntimeError("Model not fitted")
        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self._model(X_t)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def save(self, model_dir: Path) -> None:
        import torch
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), model_dir / "cnn3d_micro.pt")

    def load(self, model_dir: Path) -> None:
        import torch
        if self._model is None:
            raise RuntimeError("Must build model before loading weights")
        self._model.load_state_dict(torch.load(model_dir / "cnn3d_micro.pt", weights_only=True))
