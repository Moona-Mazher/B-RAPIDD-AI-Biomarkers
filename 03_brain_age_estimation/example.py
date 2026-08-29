"""
Example usage for the UxLSTM brain age regression model.

This example shows how to attach a scalar regression head to a
pretrained UxLSTM/BrainNext encoder for chronological brain age
estimation.
"""

import torch
from torch import nn

from uxlstm_downstream import (
    UxLSTMRegressor,
    freeze_encoder,
    unfreeze_encoder,
)


class DummyEncoder(nn.Module):
    """
    Minimal example encoder.

    Replace this with the pretrained UxLSTM/BrainNext encoder
    used in your experiment.
    """

    def __init__(self, feature_dim=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        self.projection = nn.Linear(
            32,
            feature_dim,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.projection(x)

        return x


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    feature_dim = 512

    # Replace DummyEncoder with the pretrained UxLSTM/BrainNext encoder.
    encoder = DummyEncoder(
        feature_dim=feature_dim
    )

    model = UxLSTMRegressor(
        encoder=encoder,
        feature_dim=feature_dim,
        dropout=0.2,
    )

    model = model.to(device)

    # Option 1: linear probing
    freeze_encoder(model)

    # Option 2: end-to-end fine-tuning
    # unfreeze_encoder(model)

    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=1e-4,
    )

    # Example batch:
    # batch size = 2
    # channels   = 1
    # image size = 128 x 128 x 128

    images = torch.randn(
        2,
        1,
        128,
        128,
        128,
        device=device,
    )

    chronological_age = torch.tensor(
        [65.0, 72.0],
        device=device,
    )

    predicted_age = model(images)

    loss = criterion(
        predicted_age,
        chronological_age,
    )

    print(
        "Predicted age:",
        predicted_age.detach().cpu().numpy(),
    )

    print(
        "Chronological age:",
        chronological_age.detach().cpu().numpy(),
    )

    print(
        "Loss:",
        loss.item(),
    )


if __name__ == "__main__":
    main()
