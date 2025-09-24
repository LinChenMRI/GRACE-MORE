# Golden-angle RAdial CEST MRI with MOtion-REsolved reconstruction (GRACE-MORE)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17176815.svg)](https://doi.org/10.5281/zenodo.17176815)

**Authors:** Yitian Fan, Lin Chen*  
**Email:** chenlin21@xmu.edu.cn | chenlin0430@163.com  
**Affiliation:** Department of Electronic Science, Xiamen University, Xiamen, Fujian, China  

**Python version:** 3.11.9  

---

## Overview

This repository provides code and demo for:

**Fan Y, Chen Z, Jiang X, Bie C, Meng Y, Li Y, Zhou Y, Li M, Chen L\*.**  
*GRACE-MORE: Motion-Resolved Golden-Angle Radial CEST MRI for Free-Breathing Abdominal Imaging.*

Additionally, we provide supplementary materials on representative respiratory signals extracted using AP-PCA from in vivo human abdominal CEST data.

---

## Data Access

Due to file size limitations on GitHub, the complete raw and processed datasets are hosted on **Zenodo**:

- **Zenodo record:** [https://doi.org/10.5281/zenodo.17176815](https://doi.org/10.5281/zenodo.17176815)

**Included files:**
- `rawdata.mat` – Fully sampled k-space data  
- `smap.mat` – Coil sensitivity maps  

---

## Repository Structure

```text
GRACE-MORE/
├─ data/                              # Example data (full dataset on Zenodo)
│  └─ liver/
│     ├─ full_sampling.mat            # Example fully sampled image
│     ├─ Mask.mat                     # Mask for evaluation
│     ├─ freq.mat                     # Wavelet Energy 
│     └─ traj.mat                     # Trajectory (angles)
│
├─ functions/                         # Core functions and reconstruction models
│  ├─ LS.py                           # Low-rank + Sparse operators
│  ├─ Toolbox.py                      # NUFFT operator and utility functions
│  ├─ ULS_net.py                      # ULS-Net implementation
│  ├─ Unet_ECA.py                     # 3D UNet with ECA
│  └─ EMD_CWT.m                       # Respiratory-phase binning
│
├─ model/                             # Pretrained weights
│  └─ ULS_net/
│     └─ net_params.pkl
│
├─ results/                           # Example results
│  └─ liver_GRACE_MORE.mat
│
├─ Demo.py                            # End-to-end demo script
├─ requirements.txt                   # Python dependencies
├─ README.md
└─ supplementary.(pdf|docx)           # Supplementary materials
```
---
## Usage

Runs end-to-end using the bundled example data in `./data/liver` and saves outputs to `./results`:
```bash
python Demo.py
```
---
## Citation

If you use this code or dataset, please cite:

**Dataset (Zenodo):**  
[https://doi.org/10.5281/zenodo.17176815](https://doi.org/10.5281/zenodo.17176815)

```bibtex
@dataset{gracemore,
  title        = {GRACE-MORE: Motion-Resolved Golden-Angle Radial CEST MRI for Free-Breathing Abdominal Imaging},
  author       = {Fan, Yitian and Chen, Lin and others},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17176815}
}
```
---
## Contact

We welcome your comments and suggestions.

**Last updated:** Sep 25, 2025
