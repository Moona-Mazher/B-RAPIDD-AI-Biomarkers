"""
Multimodal MRI preprocessing using MRIPreprocessor.

This script preprocesses T1-weighted, T2-weighted, and FLAIR MRI
for a single subject using the MRIPreprocessor package.

The pipeline uses T1-weighted MRI as the reference image and performs
co-registration, transformation to MNI space, skull stripping, and
cropping.

MRIPreprocessor:
https://github.com/ReubenDo/MRIPreprocessor

Example
-------
python preprocess_multimodal_mri.py \
    --t1 /path/to/T1.nii.gz \
    --t2 /path/to/T2.nii.gz \
    --flair /path/to/FLAIR.nii.gz \
    --output /path/to/output \
    --subject-id subject001
"""

import argparse
from pathlib import Path

from MRIPreprocessor.mri_preprocessor import Preprocessor


def preprocess_subject(
    t1_path: Path,
    t2_path: Path,
    flair_path: Path,
    output_dir: Path,
    subject_id: str,
):
    """
    Preprocess T1-weighted, T2-weighted, and FLAIR MRI.

    Parameters
    ----------
    t1_path : Path
        Path to the T1-weighted NIfTI image.

    t2_path : Path
        Path to the T2-weighted NIfTI image.

    flair_path : Path
        Path to the FLAIR NIfTI image.

    output_dir : Path
        Directory where preprocessing outputs will be stored.

    subject_id : str
        Subject identifier used as the output filename prefix.
    """

    input_images = {
        "T1": str(t1_path),
        "T2": str(t2_path),
        "FLAIR": str(flair_path),
    }

    for modality, image_path in input_images.items():
        if not Path(image_path).exists():
            raise FileNotFoundError(
                f"{modality} image not found: {image_path}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Processing subject: {subject_id}")
    print(f"T1:    {t1_path}")
    print(f"T2:    {t2_path}")
    print(f"FLAIR: {flair_path}")

    preprocessor = Preprocessor(
        input_images,
        output_folder=str(output_dir),
        reference="T1",
        label=None,
        prefix=f"{subject_id}_",
        already_coregistered=False,
        mni=True,
        crop=True,
    )

    preprocessor.run_pipeline()

    print(
        f"Preprocessing completed for subject: {subject_id}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess T1-weighted, T2-weighted, and FLAIR MRI "
            "using MRIPreprocessor."
        )
    )

    parser.add_argument(
        "--t1",
        type=Path,
        required=True,
        help="Path to the T1-weighted NIfTI image.",
    )

    parser.add_argument(
        "--t2",
        type=Path,
        required=True,
        help="Path to the T2-weighted NIfTI image.",
    )

    parser.add_argument(
        "--flair",
        type=Path,
        required=True,
        help="Path to the FLAIR NIfTI image.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for preprocessing outputs.",
    )

    parser.add_argument(
        "--subject-id",
        type=str,
        required=True,
        help="Subject identifier used as output prefix.",
    )

    args = parser.parse_args()

    preprocess_subject(
        t1_path=args.t1,
        t2_path=args.t2,
        flair_path=args.flair,
        output_dir=args.output,
        subject_id=args.subject_id,
    )


if __name__ == "__main__":
    main()
