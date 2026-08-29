"""
Training and evaluation utilities for chronological brain age estimation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from torch import nn


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
) -> float:
    """
    Train the regression model for one epoch.

    Returns
    -------
    float
        Mean training loss.
    """

    model.train()

    running_loss = 0.0
    n_batches = 0

    for images, ages in dataloader:
        images = images.to(device)
        ages = ages.float().to(device)

        optimizer.zero_grad()

        predictions = model(images)
        loss = criterion(predictions, ages)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_brain_age(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate chronological brain age predictions.

    Metrics
    -------
    MAE
        Mean absolute error between predicted and chronological age.

    Pearson r
        Correlation between predicted and chronological age.

    Mean brain age gap
        Mean difference between predicted and chronological age.
    """

    model.eval()

    chronological_ages = []
    predicted_ages = []

    for images, ages in dataloader:
        images = images.to(device)

        predictions = model(images)

        predicted_ages.extend(
            predictions.detach().cpu().numpy().tolist()
        )

        chronological_ages.extend(
            ages.detach().cpu().numpy().tolist()
        )

    y_true = np.asarray(chronological_ages, dtype=float)
    y_pred = np.asarray(predicted_ages, dtype=float)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    if len(y_true) > 1:
        pearson_r, _ = pearsonr(
            y_true,
            y_pred,
        )
    else:
        pearson_r = float("nan")

    brain_age_gap = y_pred - y_true

    metrics = {
        "mae": float(mae),
        "pearson_r": float(pearson_r),
        "mean_brain_age_gap": float(
            np.mean(brain_age_gap)
        ),
        "std_brain_age_gap": float(
            np.std(brain_age_gap)
        ),
    }

    return metrics


@torch.no_grad()
def predict_brain_age(
    model: nn.Module,
    dataloader,
    device: torch.device,
):
    """
    Generate chronological and predicted brain ages.

    Returns
    -------
    chronological_ages : numpy.ndarray

    predicted_ages : numpy.ndarray

    brain_age_gap : numpy.ndarray
        Predicted age minus chronological age.
    """

    model.eval()

    chronological_ages = []
    predicted_ages = []

    for images, ages in dataloader:
        images = images.to(device)

        predictions = model(images)

        predicted_ages.extend(
            predictions.detach().cpu().numpy().tolist()
        )

        chronological_ages.extend(
            ages.detach().cpu().numpy().tolist()
        )

    chronological_ages = np.asarray(
        chronological_ages,
        dtype=float,
    )

    predicted_ages = np.asarray(
        predicted_ages,
        dtype=float,
    )

    brain_age_gap = (
        predicted_ages - chronological_ages
    )

    return (
        chronological_ages,
        predicted_ages,
        brain_age_gap,
    )
