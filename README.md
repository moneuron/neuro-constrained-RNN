# Embedding Neuronal Constraints Drive Superior Learning in Recurrent Neural Networks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1frTPKt8WgwJO8FUZD0UmOM5MPmvvOXbs?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-D12424?logo=arxiv)](https://arxiv.org/abs/X)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)

> A novel approach to recurrent neural networks that incorporates biological constraints from real neural connectomes to achieve superior learning performance and brain-like network properties.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Configuration Parameters](#configuration-parameters)
  - [Model Variants](#model-variants)
  - [Cognitive Tasks](#cognitive-tasks)
- [MICrONS Dataset](#microns-dataset)
- [Results and Metrics](#results-and-metrics)
- [Extra Utilities](#extra-utilities)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Overview

This repository contains the implementation for the research paper **"Embedding Neuronal Constraints Drive Superior Learning in Recurrent Neural Networks"** by [Mo Shakiba](https://github.com/moneuron/), [Rana Rokni](https://github.com/ranarokni), [Mohammad Mohammadi](https://github.com/mmohammadi9812), and [Nima Dehghani](https://github.com/neurovium).

### The Problem

Traditional artificial recurrent neural networks (RNNs) are often trained without considering the structural and functional constraints that govern biological neural networks. This can lead to models that, while functional, lack the efficiency and network properties observed in real brains.

### Our Solution

We demonstrate that **incorporating biological constraints from real neural connectomes** into RNN architectures leads to:
- ✅ **Improved task performance** on decision-making tasks
- ✅ **Emergent brain-like network properties** (modularity, small-worldness)
- ✅ **More realistic neural dynamics**

The approach uses anatomical and functional data from the [MICrONS dataset](https://www.microns-explorer.org/) to initialize weights and apply custom regularization during training.

---

## Key Features

- 🧠 **Biological Weight Initialization**: Initialize RNN weights using real neural connectome data (STTC, correlation, or precision matrices)
- 🎯 **Three Cognitive Tasks**: Evaluate models on One-Choice Inference, Go/NoGo, and Perceptual Decision-Making tasks
- 📊 **Comprehensive Metrics**: Track accuracy, modularity, small-worldness, assortativity, and entropy
- 🔧 **11 Model Variants**: Compare biologically-constrained models against random baselines
- 🌐 **Spatial Regularization**: Apply Earth Mover's Distance and communicability regularization
- 📈 **Extensive Analysis**: Automated result saving with statistics across multiple simulation runs
- ☁️ **Cloud-Ready**: Fully runnable in Google Colab with automatic dependency installation

---

## Repository Structure

```
neuro-rnn/
├── README.md                          # This file
├── neuro-rnn.ipynb                    # Main experimental notebook
├── MICrONS-Data/
│   ├── DATA.zip                       # Preprocessed MICrONS dataset (via Git LFS)
│   └── INF.md                         # Dataset information (19,178 neurons)
└── extra/
    ├── Access-MICrONS-Data/
    │   └── MICrONS-SAVE.ipynb        # Data extraction from MICrONS database
    ├── NDA-MICrONS-Data/
    │   └── NDA.ipynb                  # Neural data analysis (STTC, correlation, precision)
    └── Quick-Plotting/
        └── plotting.ipynb             # Visualization utilities
```

### Main Components

- **`neuro-rnn.ipynb`**: The primary notebook containing all experiments from the paper
- **`MICrONS-Data/`**: Contains preprocessed neural data from the MICrONS project
  - Connectomes, spike time tiling coefficients (STTC), correlation matrices, precision matrices
  - Neuronal positions and identifiers
  - Covers 19,178 neurons across multiple sessions, scans, and fields
- **`extra/`**: Utility notebooks for data processing and visualization
  - Data extraction tools
  - Statistical analysis pipelines
  - Quick plotting functions

---

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended for larger networks)
- **Storage**: ~2GB for data and results
- **GPU**: Optional but recommended for faster training (CUDA-compatible)

### Environment Options

1. **Google Colab** (Recommended for beginners):
   - No setup required
   - Free GPU access
   - Click the "Open in Colab" badge above

2. **Local Jupyter**:
   - Requires Jupyter Notebook or JupyterLab installation
   - Better for offline work and customization

3. **Python Environment**:
   - Works with virtual environments, conda, or system Python

---

## Quick Start

### Option 1: Google Colab (Easiest)

1. Click the **"Open in Colab"** badge at the top of this README
2. Run all cells sequentially (Runtime → Run all)
3. Results will be saved and downloadable as a ZIP file

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/moneuron/neuro-rnn.git
cd neuro-rnn

# Install Jupyter (if not already installed)
pip install jupyter

# Launch the notebook
jupyter notebook neuro-rnn.ipynb

# Run all cells in order
```

**Note**: The notebook automatically installs required dependencies when first run. No manual package installation needed!

---

## Detailed Usage

### Configuration Parameters

The notebook can be customized by modifying parameters in the configuration cell:

#### Data Selection
```python
session_scan_field = '6_6_2'    # MICrONS data identifier
number_of_nodes = 312            # Number of neurons in the network
network_grid = (12, 13, 2)       # Spatial grid dimensions for random models
```

#### Weight Initialization
```python
sign_constraint = False          # Enforce positive weights (Dale's principle)
use_only_sttc = False           # Use only STTC (vs STTC + correlation)
use_precision_matrix = False    # Use precision matrix instead of correlation
```

#### Training Configuration
```python
simulations = 20                 # Number of independent runs per model
TASK1, TASK2, TASK3 = True, True, True  # Enable/disable tasks
noise_level = 0.05              # Input noise standard deviation
regu_strength = 0.3             # Spatial regularization strength (λ)
emd_strength = 0.1              # Earth Mover's Distance strength
activation_function = 'relu'    # RNN activation ('relu', 'tanh', etc.)
random_network_initialization = 'Orthogonal'  # Weight init for random models
```

### Model Variants

The notebook trains and evaluates **11 different model variants**:

| Model | Description |
|-------|-------------|
| **Random** | Baseline with orthogonal initialization |
| **Bio** | Weights from biological connectome |
| **Bio+EMD** | Bio + Earth Mover's Distance regularization |
| **Bio+C** | Bio + Communicability regularization |
| **Bio+EMD+C** | Bio + both EMD and Communicability |
| **D** | Distance-based random spatial network |
| **D+EMD** | D + Earth Mover's Distance regularization |
| **D+C** | D + Communicability regularization |
| **D+EMD+C** | D + both regularizations |
| **Dense** | Fully connected with random weights |
| **Dense+C** | Dense + Communicability regularization |

Each model is trained multiple times (default: 20 simulations) to ensure statistical robustness.

### Cognitive Tasks

#### Task 1: One-Choice Inference
A simple decision-making task where the network must choose between two options based on a single input cue.
- **Purpose**: Test basic decision-making capability
- **Difficulty**: Low

#### Task 2: Go/NoGo
The network must learn to respond (Go) or withhold response (NoGo) based on input stimulus.
- **Purpose**: Test impulse control and selective responding
- **Difficulty**: Medium

#### Task 3: Perceptual Decision-Making
A parametric coherent motion task with varying difficulty levels (5 coherence levels: 0%, 6.4%, 12.8%, 25.6%, 51.2%).
- **Purpose**: Test perceptual integration and confidence-based decisions
- **Difficulty**: High (variable)
- **Duration**: 31 time steps (fixation → stimulus → delay → response)

---

## MICrONS Dataset

### What is MICrONS?

The [MICrONS (Machine Intelligence from Cortical Networks)](https://www.microns-explorer.org/) dataset is a large-scale project that combines:
- **Connectomics**: Physical synaptic connections between neurons
- **Functional data**: Neural activity during visual stimulus presentation
- **Anatomical data**: 3D positions and morphologies of neurons

### Available Data

This repository uses preprocessed data from the MICrONS dataset:

- **Total neurons available**: 19,178 across 52 session/scan/field combinations
- **Default configuration**: Session 6, Scan 6, Field 2 (312 neurons)
- **Data types**:
  - `connectome_*.csv`: Synaptic connectivity matrix
  - `sttc_matrix.npy`: Spike Time Tiling Coefficient matrix
  - `corr_matrix.npy`: Pearson correlation matrix
  - `precision_matrix.npy`: Inverse covariance matrix
  - `positions_*.npy`: 3D coordinates of neurons
  - `spikes_*.npy`: Recorded spike trains

See [`MICrONS-Data/INF.md`](MICrONS-Data/INF.md) for a complete list of available configurations.

### Data Processing

The extra notebooks provide tools for:
1. **Data Extraction** (`extra/Access-MICrONS-Data/`): Query and download from MICrONS database
2. **Statistical Analysis** (`extra/NDA-MICrONS-Data/`): Compute STTC, correlations, and precision matrices
3. **Visualization** (`extra/Quick-Plotting/`): Generate plots and network diagrams

---

## Results and Metrics

### Output Structure

After running the notebook, results are saved in a folder named based on your configuration (e.g., `6_6_2_sttc-corr/`):

```
6_6_2_sttc-corr/
├── config.txt                      # Configuration parameters
├── TASK1_accuracy.csv             # Raw accuracy data
├── TASK1_modularity.csv           # Raw modularity data
├── TASK1_smallworld.csv           # Raw small-worldness data
├── ... (similar files for TASK2, TASK3, and other metrics)
├── summary_means.csv              # Aggregated means across simulations
├── summary_stds.csv               # Standard deviations
└── results.zip                    # All results bundled for download
```

### Tracked Metrics

For each model and task:
- **Accuracy**: Task performance (% correct)
- **Modularity**: Network community structure (Newman's Q)
- **Small-worldness**: Balance of clustering and path length (σ)
- **Assortativity**: Degree correlation (r)
- **Entropy**: Weight distribution diversity (H)

Metrics are computed at multiple training epochs and averaged across simulations.

---

## Extra Utilities

### Data Extraction (`extra/Access-MICrONS-Data/`)

Use `MICrONS-SAVE.ipynb` to:
- Query the MICrONS database directly
- Extract connectome and functional data for specific neurons
- Generate custom datasets for your own experiments

### Neural Data Analysis (`extra/NDA-MICrONS-Data/`)

Use `NDA.ipynb` to:
- Compute spike time tiling coefficients (STTC)
- Calculate Pearson correlation matrices
- Derive precision (inverse covariance) matrices
- Visualize functional connectivity

### Visualization (`extra/Quick-Plotting/`)

Use `plotting.ipynb` to:
- Generate quick plots of results
- Visualize network structures
- Create publication-ready figures

---

## Troubleshooting

### Common Issues

**Q: "ModuleNotFoundError" when running the notebook**
- A: The notebook should auto-install dependencies. If not, manually run:
  ```python
  !pip install bctpy tensorflow scipy matplotlib numpy pandas seaborn tqdm networkx
  ```

**Q: "Out of memory" errors during training**
- A: Reduce `number_of_nodes`, decrease `simulations`, or use a system with more RAM
- Alternative: Enable GPU acceleration in Colab (Runtime → Change runtime type → GPU)

**Q: Data download fails**
- A: The MICrONS data is hosted via Git LFS. If using local installation, ensure Git LFS is installed:
  ```bash
  git lfs install
  git lfs pull
  ```

**Q: Results differ from paper**
- A: Ensure you're using the same parameters as specified in the paper
- Neural network training involves randomness; results may vary slightly between runs
- Increase `simulations` for more stable statistics

**Q: Notebook runs slowly**
- A: Training 11 models × 3 tasks × 20 simulations takes time (several hours)
- Use GPU acceleration (Colab provides free GPUs)
- Reduce `simulations` for faster testing (e.g., 5 instead of 20)

### Getting Help

If you encounter issues not covered here:
1. Check the [Issues](https://github.com/moneuron/neuro-rnn/issues) page
2. Open a new issue with:
   - Your configuration parameters
   - Full error message
   - Python/TensorFlow versions
   - Execution environment (Colab/local)

---

## Contributing

We welcome contributions! Here's how you can help:

### Types of Contributions

- 🐛 **Bug Reports**: Found a bug? Open an issue with details
- 💡 **Feature Requests**: Have an idea? Suggest it in the issues
- 📝 **Documentation**: Improve README, add examples, or clarify code
- 🔬 **New Experiments**: Extend the work with new tasks or analyses
- 🛠️ **Code Improvements**: Optimize performance or add features

### Contribution Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Add comments for complex logic
- Update documentation for new features
- Include docstrings for new functions

---

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{shakiba2024embedding,
  title={Embedding Neuronal Constraints Drive Superior Learning in Recurrent Neural Networks},
  author={Shakiba, Mo and Rokni, Rana and Mohammadi, Mohammad and Dehghani, Nima},
  journal={arXiv preprint arXiv:X},
  year={2024}
}
```

**Note**: The arXiv link will be updated upon paper publication.

---

## Authors

- **[Mo Shakiba](https://github.com/moneuron/)** - Lead Developer & Research
- **[Rana Rokni](https://github.com/ranarokni)** - Research & Analysis
- **[Mohammad Mohammadi](https://github.com/mmohammadi9812)** - Research & Implementation
- **[Nima Dehghani](https://github.com/neurovium)** - Principal Investigator

---

## License

This project is open source. Please check with the authors for specific licensing terms.

### Acknowledgments

- MICrONS Consortium for the neural data
- Allen Institute for Brain Science
- IARPA MICrONS program

---

## Additional Resources

- 📄 [Paper](https://arxiv.org/abs/X) (arXiv preprint)
- 🌐 [MICrONS Project](https://www.microns-explorer.org/)
- 🧠 [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/)
- 🤖 [TensorFlow Documentation](https://www.tensorflow.org/)

---

**Happy Experimenting! 🚀**

For questions, suggestions, or collaboration opportunities, feel free to reach out through GitHub issues or contact the authors directly.
