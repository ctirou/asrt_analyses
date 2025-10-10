# 🧠 Code and data corresponding to [this manuscript](https://doi.org/10.1101/2025.08.18.670891)

Learning regularities in noise engages both neural predictive activity and representational changes
===================================================================================================
Coumarane Tirou, Oussama Abdoun, Teodóra Vékony, Laure Tosatto, Andrea Brovelli, Marine Vernet, Dezső Németh & Romain Quentin

Abstract
========
The ability to extract structured sensory patterns from a noisy environment is fundamental to cognition, yet how the brain learns complex regularities remains unclear. Using magnetoencephalography during a visuomotor task, we tracked the neural dynamics as humans learned non-adjacent temporal dependencies embedded in noise. We reveal that learning is supported by two temporally dissociable mechanisms. Neural predictive activity emerged rapidly, with stimulus-specific patterns appearing before stimulus onset and preceding measurable behavioral improvements. This is followed by a slower build-up of representational change, characterized by an increased neural pattern similarity between statistically dependent, non-adjacent elements. Both processes are supported by a distributed consortium of networks, with the sensorimotor and dorsal attentional networks playing a central role. These findings suggest that both neural predictive activity and representational changes contribute to learning regularities, revealing a temporal hierarchy in which neural predictive activity precedes behavioral improvement and is followed by neural representational changes, possibly facilitating the gradual consolidation of knowledge into stable neural representations.

## Table of Contents
- [Data](#-data)
- [How to use](#-how-to-use)
- [License](#-license)

## 🚀 Data

Raw data will be made publicly available on a hosting platform prior to publication. 

## 👨‍💻 How to use

1. Clone the repository and setup a python environment with the required libraries in the `environment.yml`.

2. Change the all paths in the `config.py` file to point to the location of the raw data and where you want to save the output data and figures. 

3. Execute the python scripts in an interactive python environment (e.g. Jupyter Notebook or Jupyter Lab) to preprocess the data, run the decoding analyses, and compute the representational similarity analyses.

4. We recommend using the RStudio for running the .qmd files.

Overall, the current scripts remain designed for research purposes, and could therefore be improved and clarified. If you judge that some codes would benefit from specific clarifications do not hesitate to contact us.

## 📃 License

BSD 3-Clause License