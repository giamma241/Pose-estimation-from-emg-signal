# Continuous Hand Pose Estimation from sEMG Signals

This repository contains our team’s implementation for the INFO-F-422 course project at ULB, which focuses on predicting continuous hand articulation using surface electromyography (sEMG) data. The aim is to develop machine learning models capable of estimating 3D joint angles of the hand in real time, for both guided and free hand gestures, using regression techniques. The source code is organized for modularity and experimentation, with support for domain adaptation, ensemble methods, and deep learning models.

## Project Description

Myoelectric prostheses rely on decoding muscle activity to control hand movements. We use high-frequency sEMG recordings from 8 electrodes and synchronized motion capture data (51 joint angles) to train and evaluate regression models. The project explores both classical and advanced machine learning methods including:

- Time-domain feature extraction
- Riemannian geometry via covariance matrices
- Neural networks
- Ensemble learning

## Project Architecture

The structure is crafted to guide every data science professional and enthusiast through a streamlined workflow, right from raw data ingestion to deployment of the final model API. The project's folder structure would look like this:

```bash
.
├── autoencoders
├── competition
│   ├── best_freemoves.npy
│   ├── best_guided.npy
│   ├── csv_generator.ipynp
│   ├── prediction_generator_freemoves.ipynb
│   └── prediction_generator_guided.ipynb
├── config
│   ├── autoencoders.py
│   ├── models.py
│   ├── regressors.py
│   ├── transformers.py
│   ├── utils.py
│   └── validation.py
├── info
├── notebooks
├── TOBEDELETED
├── visualization_unity
│   ├── working_example_freemoves.py
│   └── working_example_guided.py
└── README.md
```

### Source Code Structure

All code resides in the `src/` directory:

- **`autoencoders.py`**  
  Defines a compact autoencoder per bone, along with helper functions to encode and decode poses.  
  - Training uses early stopping and stores per-bone weights.  
  - Encoding/decoding processes iterate over 17 bones to reconstruct a compressed 21-dimensional representation.
- **`models.py`**  
  Implements various PyTorch models:  
  - Convolutional feature extractors  
  - Regression heads  
  - Domain Adversarial Network (DANN) with gradient reversal layer  
  - Trainer wrapper for adversarial training and evaluation
- **`regressors.py`**  
  Contains ensemble models and general training utilities:  
  - `VotingRegressor`, `StackingRegressor`, and a custom `NewStackingRegressor`  
  - `NNRegressor`: a general wrapper to train any PyTorch model with early stopping
- **`transformers.py`**  
  A collection of `scikit-learn`-style transformers for EMG preprocessing and feature extraction:  
  - Time-domain features  
  - Delta and wavelet features  
  - Feature selectors  
  - Session-wise transformers for applying preprocessing pipelines per session
- **`utils.py`**  
  Utility functions for:  
  - Saving predictions  
  - Plotting EMG signals  
  - Sampling neural network architectures (for architecture search)
- **`validation.py`**  
  Tools for evaluation and cross-validation:  
  - Metrics such as RMSE and NMSE  
  - Validation routines for both neural networks and general pipelines
- `notebooks/` – Example workflows demonstrating end-to-end pipelines
- `visualizations_unity/` – Unity streaming scripts:
  - `working_example_freemoves.py`
  - `working_example_guided.py`
  - These stream predicted poses to Unity at 50Hz for real-time visualization
- `csv/` – Example CSV predictions and outputs from models

## Getting Started

1. **Explore the Notebooks**  
   Go through the example workflows in `notebooks/` to understand how to chain together preprocessing, modeling, and evaluation steps.
2. **Unity Streaming**  
   Use `visualizations_unity/working_example_freemoves.py` for a minimal example of pose streaming in Unity.
3. **Have Fun** 

## Contributing Members

A group collaboration of:
- **[Marco Usula](https://github.com/@MU849)**
- **[Thomas Bellanger](https://github.com/@SMT395)**
- **[Gianmaria Brianese](https://github.com/@GB241)**

We greatly value collaboration and believe that collective insights drive optimal outcomes. If you're considering enhancing this repository, be it through bug fixes, feature suggestions, or documentation refinements, your expertise is invaluable to us.

## References
Please find here the list of all references used.

GitHub Repositories:
- https://github.com/cedricsimar/LiveHandPoseVisualisation

Books:
- https://www.researchgate.net/publication/242692234_Statistical_foundations_of_machine_learning_the_handbook

Articles:
- https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1329411/full