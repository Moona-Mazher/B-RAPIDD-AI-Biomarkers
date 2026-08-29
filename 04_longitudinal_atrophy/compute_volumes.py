"""
Extract volumetric measures from a brain segmentation map.

The script computes:

- Whole-brain parenchymal volume
- Hippocampal volume
- Ventricular volume

Whole-brain parenchymal volume excludes CSF and ventricular compartments.

Label definitions are supplied through command-line arguments so that the
script can be adapted to the segmentation label convention used by the model.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_labels(values):
    """
    Convert a list of label values to integers.
    """
    return [int(value) for value in values]


def compute_volume(segmentation, labels, voxel_volume):
    """
    Compute volume for a set of segmentation labels.

    Parameters
    ----------
    segmentation : numpy.ndarray
        Integer-valued segmentation map.

    labels : list[int]
        Labels to include in the volumetric measure.

    voxel_volume : float
        Volume of one voxel in mm^3.

    Returns
    -------
    float
        Volume in mm^3.
    """

    mask = np.isin(segmentation, labels)

    number_of_voxels = np.count_nonzero(mask)

    return float(number_of_voxels * voxel_volume)


def extract_volumes(
    segmentation_path,
    whole_brain_labels,
    hippocampus_labels,
    ventricle_labels,
):
    """
    Extract longitudinal volumetric measures from a segmentation map.
    """

    image = nib.load(str(segmentation_path))

    segmentation = np.asarray(
        image.get_fdata(),
        dtype=np.int32,
    )

    voxel_sizes = image.header.get_zooms()[:3]

    voxel_volume = float(
        np.prod(voxel_sizes)
    )

    whole_brain_volume = compute_volume(
        segmentation,
        whole_brain_labels,
        voxel_volume,
    )

    hippocampal_volume = compute_volume(
        segmentation,
        hippocampus_labels,
        voxel_volume,
    )

    ventricular_volume = compute_volume(
        segmentation,
        ventricle_labels,
        voxel_volume,
    )

    return {
        "whole_brain_parenchyma_mm3": whole_brain_volume,
        "hippocampus_mm3": hippocampal_volume,
        "ventricles_mm3": ventricular_volume,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract whole-brain parenchymal, hippocampal, "
            "and ventricular volumes from a segmentation map."
        )
    )

    parser.add_argument(
        "--segmentation",
        type=Path,
        required=True,
        help="Path to the anatomical segmentation NIfTI file.",
    )

    parser.add_argument(
        "--whole-brain-labels",
        nargs="+",
        required=True,
        help=(
            "Segmentation labels defining whole-brain parenchyma. "
            "CSF and ventricular labels should not be included."
        ),
    )

    parser.add_argument(
        "--hippocampus-labels",
        nargs="+",
        required=True,
        help="Labels corresponding to the hippocampus.",
    )

    parser.add_argument(
        "--ventricle-labels",
        nargs="+",
        required=True,
        help="Labels corresponding to the ventricular compartments.",
    )

    args = parser.parse_args()

    volumes = extract_volumes(
        segmentation_path=args.segmentation,
        whole_brain_labels=parse_labels(
            args.whole_brain_labels
        ),
        hippocampus_labels=parse_labels(
            args.hippocampus_labels
        ),
        ventricle_labels=parse_labels(
            args.ventricle_labels
        ),
    )

    print("\nExtracted volumes")

    print(
        f"Whole-brain parenchyma: "
        f"{volumes['whole_brain_parenchyma_mm3']:.2f} mm³"
    )

    print(
        f"Hippocampus: "
        f"{volumes['hippocampus_mm3']:.2f} mm³"
    )

    print(
        f"Ventricles: "
        f"{volumes['ventricles_mm3']:.2f} mm³"
    )


if __name__ == "__main__":
    main()
