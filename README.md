<img src="images/DICsuite-Logo.png" width="250" title="DICsuite Logo" alt="Logo" align="centre" vspace = 25>


![GitHub License](https://img.shields.io/github/license/cunningbio/dicsuite)
![Version](https://img.shields.io/github/v/tag/cunningbio/dicsuite?label=version)
![Last Commit](https://img.shields.io/github/last-commit/cunningbio/dicsuite)

# DICsuite
### A Python package for DIC-to-QPI reconstruction

This package provides functonality for reconstructing a quantitative estimation of phase from DIC microscopy.
In its current state, this is implemented by  shear angle estimation and inverse filtering-based reconstruction.
Logging and QC interfaces track computed shear angles, as well as a variety of quality metrics to assist users in selecting optimal input parameters for reconstruction. 

Future updates are to come to incorporate the following features:

- Additional reconstruction/contrast adjustment methods
- CellPose interfacing for DIC-powered cell segmentation
- Deep learning-powered reconstruction algorithms
---
#### Quickstart
To run a sample reconstruction using the provided agar bead image:

```bash
cd examples/
python run_recon.py
```

#### Requirements
DICsuite requires Python ≥3.8 and <3.12. Compatibility with Python 3.12+ is currently limited by external dependencies.
To install dependencies:

```bash
pip install -r requirements.txt
```

#### Optional GPU Support
If you want to use GPU acceleration, install CuPy separately:

```bash
pip install cupy-cuda12x
```
---
#### Acknowledgements
Citation details are being finalised, pending inclusion of additonal reconstruction methodologies.

Logo image generated using the microlith MATLAB package, as described in Mehta and Oldenbourg (2014).
Citation: Mehta, S. B., & Oldenbourg, R. (2014). Image simulation for biological microscopy: microlith. Biomed Opt Express, 5(6):1822-1838.
