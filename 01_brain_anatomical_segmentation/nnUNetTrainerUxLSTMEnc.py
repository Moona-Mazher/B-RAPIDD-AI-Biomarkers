"""
Custom nnU-Net v2 trainer using the UxLSTM encoder architecture.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from torch import nn

from nnunetv2.nets.UxLSTMEnc_2d import get_uxlstm_enc_2d_from_plans
from nnunetv2.nets.UxLSTMEnc_3d import get_uxlstm_enc_3d_from_plans


class nnUNetTrainerUxLSTMEnc(nnUNetTrainer):
    """
    nnU-Net v2 trainer using the UxLSTM encoder architecture.
    """

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        spatial_dims = len(configuration_manager.patch_size)

        if spatial_dims == 2:
            return get_uxlstm_enc_2d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
            )

        if spatial_dims == 3:
            return get_uxlstm_enc_3d_from_plans(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                deep_supervision=enable_deep_supervision,
            )

        raise NotImplementedError(
            f"UxLSTMEnc supports only 2D or 3D inputs, "
            f"got {spatial_dims}D."
        )
