[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1frTPKt8WgwJO8FUZD0UmOM5MPmvvOXbs?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-D12424?logo=arxiv)](https://arxiv.org/abs/X)

# Embedding Neuronal Constraints Drive Superior Learning in Recurrent Neural Networks

This repository contains the implementation for the research paper *"Embedding Neuronal Constraints Drive Superior Learning in Recurrent Neural Networks"* by [Mo Shakiba](https://github.com/moneuron/), [Rana Rokni](https://github.com/ranarokni), [Mohammad Mohammadi](https://github.com/mmohammadi9812), and [Nima Dehghani](https://github.com/neurovium). The work explores how incorporating biological constraints from the MICrONS dataset, can enhance the performance and realism of recurrent neural networks (RNNs) on decision-making tasks.

## Overview

The core of this repository is the Jupyter notebook `neuro-rnn.ipynb`, which replicates the key experiments from the paper. It constructs and trains RNN variants constrained by anatomical and functional data from the MICrONS dataset (session 6, scan 6, field 2). These models are evaluated on three cognitive tasks:

- **One-Choice Inference**
- **Go/NoGo**
- **Perceptual Decision-Making**

The notebook initializes weights using biological priors (e.g., STTC, correlations, or precision matrices), applies spatial embedding and communicability regularization during training, and computes metrics such as accuracy, modularity, small-worldness, assortativity, and entropy. Results are saved in a structured folder for analysis, including raw data, summaries, and standard deviations.

Eleven model variants are tested demonstrating that neuronal constraints lead to improved task accuracy and emergent brain-like network properties.

## Dependencies

The notebook requires the following Python libraries (installed via `pip` in the code):

- `bctpy`, `tensorflow`, `scipy`, `matplotlib`, `numpy`, `pandas`, `seaborn`, `tqdm`, `networkx`

It also downloads the preprocessed MICrONS data (connectome, STTC/correlation/precision matrices, and neuronal coordinates) from a GitHub-hosted ZIP archive.

## Usage

1. **Run the Notebook**: Execute `neuro-rnn.ipynb` in a Jupyter environment (e.g., Google Colab or local setup). The code handles dependency installation, data download, and unzipping automatically.

2. **Key Parameters**: Customize the simulation via cell-defined variables (defaults reflect paper settings):
   - `simulations`: Number of runs per model variant (default: 20).
   - `session`, `scan`, `field`: MICrONS data identifiers (default: 6, 6, 2).
   - `number_of_nodes`: Neurons in the network (default: 312).
   - `network_grid`: Grid for random spatial models (default: (12, 13, 2)).
   - `sign_constraint`: Enforce positive weights (default: False).
   - `use_only_sttc`: Use only STTC for bio-weights (default: False).
   - `use_precision_matrix`: Use precision matrix for bio-weights (default: False).
   - `TASK1`, `TASK2`, `TASK3`: Toggle tasks (default: True for all).
   - `noise_level`: Input noise stddev (default: 0.05).
   - `regu_strength`: Regularization lambda (default: 0.3).
   - `emd_strength`: Earth Mover's Distance regularization (default: 0.1).
   - `activation_function`: RNN activation (default: 'relu').
   - `random_network_initialization`: Weight init method (default: 'Orthogonal').

3. **Execution Flow**:
   - Load the preprocessed MICrONS data.
   - Initialize biological weights (lognormal scaling, spectral radius control).
   - Generate task datasets (numpy/tf.data).
   - Train RNN variants with custom regularizers (e.g., spatial embedding, communicability).
   - Compute and save metrics across epochs and simulations.

4. **Outputs**: Results are stored in a folder named based on parameters (e.g., `6_6_2_sttc-corr`). This includes:
   - Per-task raw CSV files for metrics (e.g., accuracy, modularity).
   - Summary CSVs with means and standard deviations.
   - A ZIP archive for download.

## Citation

```
@article{,

}
```
