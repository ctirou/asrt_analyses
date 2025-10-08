## Table of Contents
- [About](#-about)
- [How to use](#-how-to-use)
- [License](#-license)

## 🚀 About

This repository contains all analyses scripts to test neural predicitive activity and representational change in a visuo-motor statistical learning task using MVPA, RSA, generalized additive modeling, and source reconstruction on MEG data, as described in [Learning regularities in noise engages both neural predictive activity and representational changes](https://www.biorxiv.org/content/10.1101/2025.08.18.670891v1) (Tirou et al., 2025 biorxiv). Raw data is available through open acesss at [link]().

# How to use

1. Download the original data from this [link]()

2. Clone the repository and setup a python environment with the required libraries in the `environment.yml`.

3. Change the all paths in the `config.py` file to point to the location of the raw data and where you want to save the output data and figures. 

4. Execute the python scripts in an interactive python environment (e.g. Jupyter Notebook or Jupyter Lab) to preprocess the data, run the decoding analyses, and compute the representational similarity analyses.

5. Run the R scripts in an interactive R environment (e.g. RStudio) to run the generalized additive models and create the figures.

## 📃 License

BSD 3-Clause License