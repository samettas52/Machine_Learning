# Machine Learning — Simple KNN Example

A small educational project demonstrating a basic implementation of the K-Nearest Neighbors (KNN) algorithm in Python using NumPy.

## Overview

This repository contains a minimal, easy-to-follow KNN classifier implementation intended for learning and demonstration purposes. The implementation uses plain Python control structures (for loops and if/else) together with NumPy for numeric arrays.

## Repository structure

- `main.py` — Example KNN implementation and demonstration
- `iris/` — (optional) folder for dataset files (if present)
- `.idea/` — IDE configuration (project files)

## What the code does

The example in `main.py`:
- Defines two feature arrays (`x1`, `x2`) and a label array (`y`) with 10 samples.
- Computes Euclidean distances from a query point (8, 4) to all samples.
- Sorts samples by distance and selects the k nearest neighbors.
- Tallies neighbor labels and prints a classification result.

Note: The script prints Turkish words:
- `iyi` — means "good" (label 1)
- `kötü` — means "bad" (label 0)

## Requirements

- Python 3.8+
- NumPy

Install NumPy with:
```bash
pip install numpy
```

## Usage

Run the example script:
```bash
python main.py
```

The script will compute distances, choose the k nearest neighbors (k is set in the script), and print the predicted class.

## Customization ideas / Improvements

- Parameterize the query point and `k` via command-line arguments or function parameters.
- Add input validation and error handling.
- Support different distance metrics (Manhattan, Minkowski).
- Implement a reusable KNN function/class and add unit tests.
- Load and evaluate on real datasets (e.g., Iris) and add performance metrics (accuracy, precision, recall).
- Add documentation and example notebooks or visualizations.

## Learning purpose

This repository is intended as a learning exercise to understand how KNN works at a low level without using a machine-learning library like scikit-learn.

## License

Use as you like for learning and experimentation. Add a license file (e.g., MIT) if you want to make permissions explicit.

## Author

samettas52 — https://github.com/samettas52
