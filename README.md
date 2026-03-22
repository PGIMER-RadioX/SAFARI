# SAFARI
SAFARI: Self-supervised Tumor Agnostic Foundation for Advanced Radiomics Integration

# Licence
[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http.creativecommons.org/licenses/by-nc-sa/4.0/).



## Project Structure
```
SAFARI/
├── .gitignore
├── LICENSE
├── README.md
├── ENV.yml
├── scripts/
│   └── train_model.py
├── setup.py
└── src/
    ├── foundation_model/
    │   ├── __init__.py
    │   ├── model.py
    │   ├── training.py
    │   └── data_utils.py

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

### 1. Pretraining the Foundation Model

To pretrain the foundation model, use the `scripts/train_model.py` script. You need to provide the path to your radiomics data (either a single CSV file or a directory of CSVs) and an output directory to save the model and artifacts.

```bash
python scripts/train_foundation_model.py --data /path/to/your/data --output-dir /path/to/save/model
```

For a full list of training options, run:
```bash
python scripts/train_foundation_model.py --help
```
