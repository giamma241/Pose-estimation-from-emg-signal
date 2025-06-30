# Continuous Hand Pose Estimation from sEMG Signals

This repository contains our team’s implementation for the INFO-F-422 course project at ULB, which focuses on predicting continuous hand articulation using surface electromyography (sEMG) data. The aim is to develop machine learning models capable of estimating 3D joint angles of the hand in real time, for both guided and free hand gestures, using regression techniques.

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