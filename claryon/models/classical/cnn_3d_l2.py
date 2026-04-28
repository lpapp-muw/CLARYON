"""3D CNN with L2 input normalization — mimics amplitude encoding's projection."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "cnn_3d_l2")
class CNN3DL2Model(ModelBuilder):
    """3D CNN with L2 input normalization (the critical classical control).

    Identical to CNN3DModel except an L2 normalization layer is prepended:
    x → x/||x|| before the first convolution. This isolates whether
    amplitude encoding's domain robustness comes from L2 projection alone.

    Expects input X of shape (N, C, D, H, W).
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 1,
        n_conv_layers: int = 3,
        base_filters: int = 8,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 4,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self._n_classes = n_classes
        self._n_channels = n_channels
        self._n_conv_layers = n_conv_layers
        self._base_filters = base_filters
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._seed = seed
        self._model: Any = None
        self._task_type = TaskType.BINARY

    @property
    def name(self) -> str:
        return "cnn_3d_l2"

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE_3D

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS)

    def _build_net(self, in_channels: int, spatial_shape: tuple[int, ...]) -> Any:
        """Build a 3D CNN with L2 input normalization prepended."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _L2InputNorm(nn.Module):
            """Normalize input volume to unit L2 norm (mimics amplitude encoding)."""

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                flat = x.view(x.size(0), -1)
                normed = F.normalize(flat, p=2, dim=1)
                return normed.view(x.shape)

        conv_layers = []
        c_in = in_channels
        c_out = self._base_filters
        for _ in range(self._n_conv_layers):
            conv_layers.extend([
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm3d(c_out),
                nn.ReLU(),
                nn.MaxPool3d(2),
            ])
            c_in = c_out
            c_out = min(c_out * 2, 64)

        dummy = torch.zeros(1, in_channels, *spatial_shape)
        conv = nn.Sequential(*conv_layers)
        with torch.no_grad():
            flat_size = conv(dummy).view(1, -1).shape[1]

        net = nn.Sequential(
            _L2InputNorm(),
            conv,
            nn.Flatten(),
            nn.Linear(flat_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, self._n_classes),
        )
        return net

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Train 3D CNN with L2 normalization.

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

        spatial_shape = X.shape[2:]
        self._model = self._build_net(X.shape[1], spatial_shape)
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
            logger.debug("cnn_3d_l2 epoch %d/%d loss=%.4f", epoch + 1, self._epochs, total_loss / len(loader))

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
        torch.save(self._model.state_dict(), model_dir / "cnn3d_l2.pt")

    def load(self, model_dir: Path) -> None:
        import torch
        if self._model is None:
            raise RuntimeError("Must build model before loading weights")
        self._model.load_state_dict(torch.load(model_dir / "cnn3d_l2.pt", weights_only=True))
