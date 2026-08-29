"""
Inference wrapper for a trained UxLSTM nnU-Net v2 segmentation model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def run_inference(
    model_folder: Path,
    input_dir: Path,
    output_dir: Path,
    folds: tuple,
    checkpoint_name: str,
    device: str,
    disable_tta: bool,
) -> None:
    """
    Run segmentation inference using a trained nnU-Net v2 model.

    Parameters
    ----------
    model_folder : Path
        Path to the trained nnU-Net model directory.

    input_dir : Path
        Directory containing input MRI volumes following nnU-Net naming
        conventions, e.g. CASE001_0000.nii.gz.

    output_dir : Path
        Directory where predicted segmentation maps will be saved.

    folds : tuple
        Fold(s) used for inference.

    checkpoint_name : str
        Name of the checkpoint file.

    device : str
        Device used for inference: cuda, cpu, or mps.

    disable_tta : bool
        Disable test-time augmentation if True.
    """

    if not model_folder.exists():
        raise FileNotFoundError(
            f"Model folder not found: {model_folder}"
        )

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    torch_device = torch.device(device)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not disable_tta,
        perform_everything_on_device=torch_device.type == "cuda",
        device=torch_device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=folds,
        checkpoint_name=checkpoint_name,
    )

    predictor.predict_from_files(
        str(input_dir),
        str(output_dir),
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


def parse_folds(values: list[str]) -> tuple:
    """
    Convert command-line fold arguments to the format expected by nnU-Net.
    """

    folds = []

    for value in values:
        folds.append(
            value if value == "all" else int(value)
        )

    return tuple(folds)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run nnU-Net v2 inference with a trained "
            "UxLSTM encoder segmentation model."
        )
    )

    parser.add_argument(
        "--model-folder",
        type=Path,
        required=True,
        help="Path to the trained nnU-Net model folder.",
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input MRI volumes.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for predicted segmentation maps.",
    )

    parser.add_argument(
        "--fold",
        action="append",
        default=None,
        help="Repeat for multiple folds; use 'all' for fold_all.",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoint_final.pth",
        help="Checkpoint filename inside the fold directory.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device used for inference.",
    )

    parser.add_argument(
        "--disable-tta",
        action="store_true",
        help="Disable mirroring/test-time augmentation.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    folds = parse_folds(
        args.fold or ["0"]
    )

    run_inference(
        model_folder=args.model_folder,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        folds=folds,
        checkpoint_name=args.checkpoint,
        device=args.device,
        disable_tta=args.disable_tta,
    )


if __name__ == "__main__":
    main()
