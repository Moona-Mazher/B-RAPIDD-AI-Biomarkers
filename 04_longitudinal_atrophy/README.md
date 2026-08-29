# Longitudinal Atrophy Analysis Task

This directory contains code for the **longitudinal atrophy analysis** evaluated in the B-RAPIDD study.

The aim of this analysis is to assess whether accelerated MRI preserves sensitivity to longitudinal structural brain changes over time.

## Workflow

Baseline and follow-up MRI scans are first processed using the anatomical segmentation pipeline provided in:

`../01_brain_anatomical_segmentation/`

The resulting segmentation maps are then used to extract predefined volumetric measures at each time point.

```text
Baseline MRI
    │
    ▼
Segmentation inference
    │
    ▼
Baseline volumetric measures

Follow-up MRI
    │
    ▼
Segmentation inference
    │
    ▼
Follow-up volumetric measures

Baseline + follow-up volumes
    │
    ▼
Annualized percentage volume change
    │
    ▼
Comparison across MRI acquisition protocols
```

## Longitudinal Measures

The longitudinal analysis focuses on three predefined volumetric measures:

- **Whole-brain parenchymal volume**
- **Hippocampal volume**
- **Ventricular volume**

### Whole-Brain Parenchymal Volume

Whole-brain volume is defined as an aggregate measure of brain parenchymal tissue rather than a simple sum of all 32 anatomical segmentation labels.

The measure includes the relevant brain tissue regions used in the study while excluding:

- Cerebrospinal fluid (CSF)
- Ventricular compartments

### Hippocampal Volume

Hippocampal volume is calculated from the corresponding hippocampal segmentation labels.

Where bilateral labels are available, left and right hippocampal volumes are combined to obtain the total hippocampal volume.

### Ventricular Volume

Ventricular volume is calculated from the ventricular segmentation labels.

Where multiple ventricular compartments are included in the analysis, their volumes are combined to obtain the total ventricular volume used for longitudinal comparison.

## Annualized Volume Change

For each subject and volumetric measure, percentage volume change is calculated relative to the baseline measurement and annualized according to the follow-up interval:

```text
Annualized change (%/year) =

    (Follow-up volume - Baseline volume)
    ------------------------------------ × 100
              Baseline volume

    ------------------------------------
       Follow-up interval in years
```

Negative values indicate volume loss, whereas positive values indicate volume expansion.

This allows whole-brain and hippocampal atrophy, as well as ventricular enlargement, to be compared across routine and accelerated MRI acquisitions.

## Input

The analysis requires:

- Baseline segmentation map
- Follow-up segmentation map
- Baseline scan date or study time point
- Follow-up scan date or study time point

The follow-up interval is used to annualize the observed volumetric change.

## Files

The longitudinal analysis code is organized as:

```text
04_longitudinal_atrophy/
├── README.md
├── compute_volumes.py
└── longitudinal_atrophy.py
```

### `compute_volumes.py`

Extracts whole-brain parenchymal, hippocampal, and ventricular volumes from anatomical segmentation maps.

### `longitudinal_atrophy.py`

Calculates percentage and annualized volume change between baseline and follow-up measurements.

## Related Study

This analysis forms part of the B-RAPIDD study:

**Accelerated MRI Preserves AI-Derived Neuroimaging Biomarkers for Dementia Diagnosis and Monitoring.**

For the complete study workflow, preprocessing information, data availability, and citation details, see the main repository README.
