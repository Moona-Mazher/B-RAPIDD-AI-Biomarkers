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

## Tasks

### 1. 32 Brain Anatomical Segmentation Task

Whole-brain anatomical segmentation was evaluated across **32 brain structures** to assess whether accelerated MRI preserves regional anatomical information required for quantitative neuroimaging analysis.

Code related to this task is provided in:

`01_anatomical_segmentation/`

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

    B-RAPIDD-AI-Biomarkers/
    ├── README.md
    ├── LICENSE
    ├── CITATION.cff
    ├── requirements.txt
    │
    ├── figures/
    │   └── Figure1.png
    │
    ├── 01_anatomical_segmentation/
    │
    ├── 02_diagnostic_classification/
    │
    ├── 03_brain_age_estimation/
    │
    └── 04_longitudinal_atrophy/

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
