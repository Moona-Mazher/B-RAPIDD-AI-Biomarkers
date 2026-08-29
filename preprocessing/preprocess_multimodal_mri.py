"""
Multimodal MRI preprocessing utility.

This script prepares T1-weighted, T2-weighted, and FLAIR NIfTI images
for preprocessing using the MRIPreprocessor pipeline.

Inputs
------
T1-weighted MRI
T2-weighted MRI
FLAIR MRI

Example
-------
python preprocess_multimodal_mri.py \
    --t1 /path/to/T1.nii.gz \
    --t2 /path/to/T2.nii.gz \
    --flair /path/to/FLAIR.nii.gz \
    --output /path/to/output
"""

import argparse
from pathlib import Path

import SimpleITK as sitk


def validate_image(image_path: Path, modality: str):
    """
    Validate that an MRI volume exists and can be read.

    Parameters
    ----------
    image_path : Path
        Path to the NIfTI image.

    modality : str
        Name of the MRI modality.

    Returns
    -------
    SimpleITK.Image
        Loaded MRI image.
    """

    if not image_path.exists():
        raise FileNotFoundError(
            f"{modality} image not found: {image_path}"
        )

    image = sitk.ReadImage(str(image_path))

    print(
        f"{modality}: {image_path.name} | "
        f"size={image.GetSize()} | "
        f"spacing={image.GetSpacing()}"
    )

    return image


def preprocess_subject(
    t1_path: Path,
    t2_path: Path,
    flair_path: Path,
    output_dir: Path,
):
    """
    Validate and preprocess multimodal MRI from one subject.

    The preprocessing pipeline expects T1-weighted, T2-weighted,
    and FLAIR NIfTI images.

    Parameters
    ----------
    t1_path : Path
        T1-weighted MRI.

    t2_path : Path
        T2-weighted MRI.

    flair_path : Path
        FLAIR MRI.

    output_dir : Path
        Directory for preprocessing outputs.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Validating MRI inputs...")

    validate_image(t1_path, "T1")
    validate_image(t2_path, "T2")
    validate_image(flair_path, "FLAIR")

    print("\nAll required MRI modalities were successfully loaded.")

    # ------------------------------------------------------------------
    # MRI preprocessing
    # ------------------------------------------------------------------
    #
    # The B-RAPIDD preprocessing workflow uses MRIPreprocessor:
    #
    # https://github.com/ReubenDo/MRIPreprocessor
    #
    # T1, T2 and FLAIR images are provided as inputs to the preprocessing
    # pipeline before downstream model inference.
    #
    # Add the MRIPreprocessor command/API used for the study here.
    #
    # ------------------------------------------------------------------

    print(
        "\nInputs are ready for the MRIPreprocessor pipeline."
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess T1-weighted, T2-weighted and FLAIR "
            "MRI volumes for downstream analysis."
        )
    )

    parser.add_argument(
        "--t1",
        type=Path,
        required=True,
        help="Path to the T1-weighted NIfTI image."
    )

    parser.add_argument(
        "--t2",
        type=Path,
        required=True,
        help="Path to the T2-weighted NIfTI image."
    )

    parser.add_argument(
        "--flair",
        type=Path,
        required=True,
        help="Path to the FLAIR NIfTI image."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for preprocessing outputs."
    )

    args = parser.parse_args()

    preprocess_subject(
        t1_path=args.t1,
        t2_path=args.t2,
        flair_path=args.flair,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
