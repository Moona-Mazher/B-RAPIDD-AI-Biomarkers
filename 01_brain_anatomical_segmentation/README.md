# 32 Brain Anatomical Segmentation Task

This directory contains code for the **32 brain anatomical segmentation task** used in the B-RAPIDD study.

The segmentation model uses the BrainNext/xLSTM-based encoder within an nnU-Net v2 segmentation framework to obtain anatomical brain segmentations from T1-weighted MRI.

The resulting segmentation maps are used to assess anatomical consistency between routine and accelerated MRI acquisitions and to support downstream volumetric analyses.

## Input

The segmentation pipeline expects a preprocessed T1-weighted MRI volume.

MRI preprocessing is described in the main repository under:

`../preprocessing/`

## Files

```text
01_brain_anatomical_segmentation/
├── README.md
├── inference.py
└── nnUNetTrainerUxLSTMEnc.py
```

### `inference.py`

Runs segmentation inference using a trained nnU-Net v2 model and produces anatomical label maps for the input T1-weighted MRI volumes.

### `nnUNetTrainerUxLSTMEnc.py`

Defines the custom nnU-Net v2 trainer incorporating the xLSTM-based encoder architecture used for the segmentation model.

## Example Inference

```bash
python inference.py \
    --model-folder /path/to/trained/model \
    --input-dir /path/to/imagesTs \
    --output-dir /path/to/predictions \
    --fold 0
```

Trained model weights are not distributed in this repository and must be supplied separately.

## Output

The inference pipeline generates multi-label anatomical segmentation maps containing the brain structures used in the study.

These segmentation outputs are subsequently used for quantitative comparison between routine and accelerated MRI and for longitudinal volumetric analysis.

## Related Study

This analysis forms part of the B-RAPIDD study:

**Accelerated MRI Preserves AI-Derived Neuroimaging Biomarkers for Dementia Diagnosis and Monitoring.**

For the complete workflow, preprocessing information, data availability, and citation details, see the main repository README.
