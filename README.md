# B-RAPIDD

**Accelerated MRI Preserves AI-Derived Neuroimaging Biomarkers for Dementia Diagnosis and Monitoring**

This repository contains code supporting the analyses performed in the B-RAPIDD study, which evaluates whether accelerated brain MRI preserves the information required for AI-derived neuroimaging biomarkers across diagnostic, quantitative, and longitudinal applications.

The study evaluates paired routine and accelerated MRI acquisitions across four clinically relevant downstream tasks.

<p align="center">
  <img src="figures/Figure1.png" width="900">
</p>

<p align="center">
  <b>Figure 1.</b> Overview of the B-RAPIDD study framework, including paired routine and accelerated MRI acquisitions and evaluation across anatomical segmentation, diagnostic classification, chronological brain age estimation, and longitudinal atrophy analysis.
</p>

---
## Preprocessing

MRI data were prepared in two stages prior to the downstream analyses:

1. **DICOM-to-NIfTI conversion**
2. **Multimodal MRI preprocessing of T1-weighted, T2-weighted, and FLAIR images**

The corresponding scripts are provided in the `preprocessing/` directory.

### DICOM to NIfTI Conversion

Raw DICOM MRI series can be converted to compressed NIfTI (`.nii.gz`) format using:

`preprocessing/dicom_to_nifti.py`

The conversion utility uses `pydicom` for DICOM handling and `dicom2nifti` for NIfTI conversion.

#### Usage

```bash
python preprocessing/dicom_to_nifti.py \
    --input /path/to/dicom_dataset \
    --output /path/to/nifti_output
```

The script searches subject directories for DICOM series and stores the converted NIfTI images in subject-specific output directories.

---

### Multimodal MRI Preprocessing

Following conversion to NIfTI format, T1-weighted, T2-weighted, and FLAIR MRI volumes are preprocessed using [MRIPreprocessor](https://github.com/ReubenDo/MRIPreprocessor).

Install MRIPreprocessor directly from its GitHub repository:

```bash
pip install git+https://github.com/ReubenDo/MRIPreprocessor#egg=MRIPreprocessor
```

The preprocessing script is:

`preprocessing/preprocess_multimodal_mri.py`

#### Usage

```bash
python preprocessing/preprocess_multimodal_mri.py \
    --t1 /path/to/T1.nii.gz \
    --t2 /path/to/T2.nii.gz \
    --flair /path/to/FLAIR.nii.gz \
    --output /path/to/output \
    --subject-id subject001
```

The script takes three MRI modalities for each subject:

- **T1-weighted MRI**
- **T2-weighted MRI**
- **FLAIR MRI**

The T1-weighted image is used as the reference image for multimodal preprocessing.

### Preprocessing Workflow

```text
Raw DICOM MRI
      │
      ▼
DICOM-to-NIfTI conversion
      │
      ▼
T1 + T2 + FLAIR NIfTI images
      │
      ▼
MRIPreprocessor
      │
      ▼
Preprocessed MRI
      │
      ├── 32 Brain Anatomical Segmentation
      ├── Diagnostic Classification
      ├── Chronological Brain Age Estimation
      └── Longitudinal Atrophy Analysis
```

### Dependencies

The preprocessing utilities require:

- `pydicom`
- `dicom2nifti`
- `SimpleITK`
- `nibabel`
- `numpy`

Install the required Python packages using:

```bash
pip install pydicom dicom2nifti SimpleITK nibabel numpy
```

Install MRIPreprocessor using:

```bash
pip install git+https://github.com/ReubenDo/MRIPreprocessor#egg=MRIPreprocessor
```

## Tasks

### 1. 32 Brain Anatomical Segmentation Task

Whole-brain anatomical segmentation was evaluated across **32 brain structures** to assess whether accelerated MRI preserves regional anatomical information required for quantitative neuroimaging analysis.

Code related to this task is provided in:

`01_brain_anatomical_segmentation/`

---

### 2. Diagnostic Classification Task

Diagnostic classification was evaluated using AI-derived representations from routine and accelerated MRI.

The analyses included:

- Disease vs non-neurodegenerative control classification
- Alzheimer’s disease vs non-neurodegenerative control classification

Diagnostic robustness was additionally assessed using an independent AI model.

Code related to this task is provided in:

`02_diagnostic_classification/`

---

### 3. Chronological Brain Age Estimation Task

Chronological brain age estimation was used to assess whether accelerated MRI preserves age-related structural information captured by the AI model.

Predicted brain age derived from routine and accelerated MRI was compared with chronological age and across acquisition protocols.

Code related to this task is provided in:

`03_brain_age_estimation/`

---

### 4. Longitudinal Atrophy Analysis Task

Longitudinal MRI was used to evaluate whether accelerated acquisitions preserve sensitivity to structural brain changes over time.

The analysis included longitudinal changes in:

- Whole-brain volume
- Hippocampal volume
- Ventricular volume

Annualized atrophy estimates derived from routine and accelerated MRI were compared across acquisition protocols.

Code related to this task is provided in:

`04_longitudinal_atrophy/`

---

## Repository Structure

```text
B-RAPIDD-AI-Biomarkers/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── figures/
│   └── Figure1.png
│
├── preprocessing/
│   ├── dicom_to_nifti.py
│   └── preprocess_multimodal_mri.py
│
├── 01_brain_anatomical_segmentation/
│
├── 02_diagnostic_classification/
│
├── 03_brain_age_estimation/
│
└── 04_longitudinal_atrophy/
```

Each task directory contains the code and task-specific documentation required for the corresponding analysis.

---

## Data Availability

The imaging data used in this study contain clinical research data and cannot be publicly released through this repository.

Access to the underlying data is subject to the relevant institutional approvals, ethical requirements, governance procedures, and data-sharing agreements.

No identifiable participant data are included in this repository.

---

## Code Availability

This repository provides code supporting the principal analyses reported in the study.

The repository is intended to facilitate methodological transparency and reproducibility. External or independently developed models are not redistributed where licensing or access restrictions prevent public release.

---

## Citation

If you use **BrainNext** or code from this repository, please cite:

**Mazher M, Qayyum A, Niederer SA, Alexander DC. BrainNext: A General-Purpose Self-Supervised Foundation Model for Brain MRI Analysis. arXiv preprint arXiv:2607.17782, 2026.**

BibTeX:

    @article{mazher2026brainnext,
      title={BrainNext: A General-Purpose Self-Supervised Foundation Model for Brain MRI Analysis},
      author={Mazher, Moona and Qayyum, Abdul and Niederer, Steven A and Alexander, Daniel C},
      journal={arXiv preprint arXiv:2607.17782},
      year={2026}
    }

---

## License

License information is provided in the `LICENSE` file.

---

## Contact

For questions related to this repository or the associated study, please open an issue through this GitHub repository.
