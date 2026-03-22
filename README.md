# SAFARI: A Self-supervised Tumor-Agnostic Framework for Radiomics Representation Learning in Oncology Practice

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/) ![Status](https://img.shields.io/badge/Paper%20Status-Under--Review-purple)

## Project Structure
```
SAFARI/
├── config
│   └── exampleCT.yaml
├── ENV.yml
├── out
│   └── models
│       └── best_model.pt
├── README.md
├── scripts
│   ├── feature_extractor.py
│   └── train_model.py
└── src
    └── foundation_model
        ├── data_utils.py
        ├── __init__.py
        ├── model.py
        └── training.py
```
## Overview

This repository provides the code for pretraining a foundation model on radiomics data from pan-organ tumors using Self-Supervised Learning Techniques. The goal is to learn meaningful representations from large-scale radiomics datasets that can be transferred to various downstream tasks.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PGIMER-RadioX/SAFARI.git
    cd SAFARI
    ```
      > If `git` is not installed, please follow [this guide](https://git-scm.com/install/).

2.  **Install a `conda` virtual environment from ENV.yml:**
    ```bash
    conda env create --f ENV.yml
    ```
      > If `conda` is not installed, please follow [this guide](https://www.anaconda.com/docs/getting-started/miniconda/install/overview).

3. **Activate `conda` enviroment**
   ```bash
   conda activate safari
   ```
4. **Intall `pytorch` with `CUDA` runtimes in `safari` env**
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

## Usage

1. Radiomic Feature Extraction

    To extract Radiomic features, run `scripts/feature_extractor.py` script. Prepare a metadata csv which acts as input for the script and place it as `data/metadata.csv`. 

    **Metadata CSV Column Structure**:
    ```csv
    patient_id,ct_scan_path,mask_path
    P001,/data/patient001/ct.nii,/data/patient001/mask.nii
    P002,/data/patient002/ct.nii,/data/patient002/mask.nii
    ...
    ```
    
    Configuration file for PyRadiomics feature extractor is already present `config/exampleCT.yaml`. This config is unmodified copy of the original config [commited](https://github.com/AIM-Harvard/pyradiomics/blob/master/examples/exampleSettings/exampleCT.yaml) in the PyRadiomics repo.
    
    Running this script will generate a `out/features.csv` file. This is a single output file where all extracted radiomic features are saved **incrementally**. 

2. Pretraining the Foundation Model

    To pretrain the foundation model, use the `scripts/train_model.py` script. You need to provide the path to your radiomic feature data (either a single CSV file or a directory of CSVs) and an output directory to save the model checkpoint.
    
    ```bash
    python scripts/train_foundation_model.py --data out/features.csv --output-dir out/model
    ```
    
    For a full list of training options, run:
    ```bash
    python scripts/train_foundation_model.py --help
    ```
