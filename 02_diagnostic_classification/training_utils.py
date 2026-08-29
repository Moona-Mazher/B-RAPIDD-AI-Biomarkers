"""
Training and evaluation utilities for downstream classification tasks.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch import nn


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Returns
    -------
    float
        Mean training loss.
    """

    model.train()

    running_loss = 0.0
    n_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_binary_classifier(
    model: nn.Module,
    dataloader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate a binary classification model.

    Metrics
    -------
    Accuracy
    Precision
    Recall
    F1 score
    AUROC
    """

    model.eval()

    all_labels = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)

        all_probs.extend(
            probs.detach().cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.detach().cpu().numpy().tolist()
        )

    y_true = np.asarray(all_labels)
    y_prob = np.asarray(all_probs)

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(
            y_true,
            y_prob,
        )
    else:
        metrics["auroc"] = float("nan")

    return metrics


@torch.no_grad()
def predict_binary_classifier(
    model: nn.Module,
    dataloader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate probabilities and binary predictions.

    Returns
    -------
    probabilities : numpy.ndarray
        Predicted positive-class probabilities.

    predictions : numpy.ndarray
        Thresholded binary predictions.
    """

    model.eval()

    probabilities = []

    for images, _ in dataloader:
        images = images.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)

        probabilities.extend(
            probs.detach().cpu().numpy().tolist()
        )

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return probabilities, predictions
