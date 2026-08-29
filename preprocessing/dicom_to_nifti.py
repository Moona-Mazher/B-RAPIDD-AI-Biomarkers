
"""
DICOM to NIfTI conversion utility.

This script converts DICOM MRI series into compressed NIfTI files
using pydicom and dicom2nifti.

Example
-------
python dicom_to_nifti.py \
    --input /path/to/dicom_dataset \
    --output /path/to/nifti_output
"""

import argparse
from pathlib import Path

import dicom2nifti
import pydicom


def find_dicom_series(subject_dir: Path):
    """
    Find directories containing DICOM (.dcm) files.

    Parameters
    ----------
    subject_dir : Path
        Path to a subject directory.

    Returns
    -------
    list of Path
        Directories containing DICOM files.
    """
    series_dirs = []

    for directory in subject_dir.rglob("*"):
        if not directory.is_dir():
            continue

        if any(directory.glob("*.dcm")):
            series_dirs.append(directory)

    return sorted(set(series_dirs))


def convert_dataset(input_dir: Path, output_dir: Path):
    """
    Convert DICOM series for all subjects in a dataset to NIfTI.

    Parameters
    ----------
    input_dir : Path
        Root directory containing subject folders with DICOM data.

    output_dir : Path
        Root directory where converted NIfTI files will be saved.
    """

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(
        directory
        for directory in input_dir.iterdir()
        if directory.is_dir()
    )

    if not subject_dirs:
        raise RuntimeError(
            f"No subject directories found in: {input_dir}"
        )

    print(f"Found {len(subject_dirs)} subject directories.")

    for subject_dir in subject_dirs:

        print(f"\nProcessing subject: {subject_dir.name}")

        subject_output = output_dir / subject_dir.name
        subject_output.mkdir(
            parents=True,
            exist_ok=True
        )

        series_dirs = find_dicom_series(subject_dir)

        if not series_dirs:
            print("  No DICOM series found.")
            continue

        print(f"  Found {len(series_dirs)} DICOM series.")

        for series_dir in series_dirs:

            dicom_files = list(series_dir.glob("*.dcm"))

            if not dicom_files:
                continue

            try:
                # Read one DICOM header to validate the series
                # and retrieve a human-readable description.
                dataset = pydicom.dcmread(
                    dicom_files[0],
                    stop_before_pixels=True
                )

                series_description = getattr(
                    dataset,
                    "SeriesDescription",
                    series_dir.name
                )

                print(f"  Converting: {series_description}")

                dicom2nifti.convert_directory(
                    str(series_dir),
                    str(subject_output),
                    compression=True,
                    reorient=True
                )

            except Exception as exc:
                print(
                    f"  Warning: failed to convert "
                    f"{series_dir}: {exc}"
                )

    print("\nDICOM to NIfTI conversion completed.")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert DICOM MRI series to compressed NIfTI files."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root directory containing subject DICOM folders."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for converted NIfTI files."
    )

    args = parser.parse_args()

    convert_dataset(
        input_dir=args.input,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
