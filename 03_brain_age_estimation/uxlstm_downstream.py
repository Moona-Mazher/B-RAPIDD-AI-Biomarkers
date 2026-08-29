"""
UxLSTM encoder wrapper for chronological brain age regression.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _select_deepest_feature(features: Any) -> torch.Tensor:
    """
    Return the deepest tensor from common encoder output structures.
    """

    if torch.is_tensor(features):
        return features

    if isinstance(features, (list, tuple)):
        tensors = [x for x in features if torch.is_tensor(x)]

        if not tensors:
            raise TypeError(
                "Encoder output list/tuple contains no tensors."
            )

        return tensors[-1]

    if isinstance(features, dict):
        for key in (
            "features",
            "out",
            "encoder_features",
            "bottleneck",
        ):
            value = features.get(key)

            if torch.is_tensor(value):
                return value

            if isinstance(value, (list, tuple)):
                return _select_deepest_feature(value)

        tensors = [
            value
            for value in features.values()
            if torch.is_tensor(value)
        ]

        if tensors:
            return tensors[-1]

    raise TypeError(
        "Unsupported encoder output type. "
        "Expected tensor, list/tuple, or dict."
    )


class GlobalPool(nn.Module):
    """
    Global average pooling for vector, 2D, or 3D feature representations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim == 2:
            return x

        if x.ndim < 3:
            raise ValueError(
                f"Expected [B, C] or a spatial feature map, "
                f"but received shape {x.shape}."
            )

        spatial_dims = tuple(range(2, x.ndim))

        return x.mean(dim=spatial_dims)


class UxLSTMFeatureExtractor(nn.Module):
    """
    Convert UxLSTM encoder outputs into a pooled feature vector.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.pool = GlobalPool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = self.encoder(x)

        features = _select_deepest_feature(features)

        return self.pool(features)


class UxLSTMRegressor(nn.Module):
    """
    UxLSTM encoder with a scalar regression head for brain age estimation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.features = UxLSTMFeatureExtractor(encoder)

        self.head = nn.Sequential(
            nn.Dropout(dropout)
            if dropout > 0
            else nn.Identity(),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = self.features(x)

        prediction = self.head(features)

        return prediction.squeeze(-1)


def freeze_encoder(model: nn.Module) -> None:
    """
    Freeze encoder parameters for linear-probe regression.
    """

    for parameter in model.features.encoder.parameters():
        parameter.requires_grad = False


def unfreeze_encoder(model: nn.Module) -> None:
    """
    Unfreeze encoder parameters for end-to-end fine-tuning.
    """

    for parameter in model.features.encoder.parameters():
        parameter.requires_grad = True
