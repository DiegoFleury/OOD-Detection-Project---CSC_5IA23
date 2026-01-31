# OOD-Detection-Project---CSC_5IA23

Deep Learning Theory - Practical Work (MVA + ENSTA)

![Training Curves](results/figures/training/training_curves.gif)
![OOD Detection](results/figures/ood_detection/roc_curves.gif)
![Neural Collapse](results/figures/neural_collapse/nc_evolution.gif)

## Overview

This project implements and analyzes Out-of-Distribution (OOD) detection methods and studies the Neural Collapse phenomenon in deep neural networks. We train a ResNet-18 classifier on CIFAR-100 and evaluate various OOD scoring methods.

## Project Structure

```
ood-neural-collapse/
├── README.md
├── report.pdf
├── notebooks/
│   ├── 01_training.ipynb
│   ├── 02_ood_detection.ipynb
│   ├── 03_neural_collapse.ipynb
│   └── 04_bonus_analysis.ipynb
├── src/
│   ├── models/
│   ├── ood_scores/
│   ├── neural_collapse/
│   ├── data/
│   └── utils/
├── configs/
├── checkpoints/
└── results/
```

## Implemented Methods

### OOD Detection Scores
- **Output-based**: MSP, Max Logit, Energy Score
- **Distance-based**: Mahalanobis Distance
- **Feature-based**: ViM, NECO

### Neural Collapse Analysis
- NC1: Within-class variance
- NC2: Class means simplex structure
- NC3: Classifier alignment
- NC4: Self-duality
- NC5: Layer-wise collapse

## Datasets

- **In-Distribution**: CIFAR-100
- **Out-of-Distribution**: SVHN, CIFAR-10, Textures

## Usage

### Training
```bash
jupyter notebook notebooks/01_training.ipynb
```

### OOD Detection
```bash
jupyter notebook notebooks/02_ood_detection.ipynb
```

### Neural Collapse Analysis
```bash
jupyter notebook notebooks/03_neural_collapse.ipynb
```

## Results

### Model Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | TBD |
| Training Time | TBD |

### OOD Detection (AUROC)
| Method | SVHN | CIFAR-10 | Textures | Average |
|--------|------|----------|----------|---------|
| MSP | TBD | TBD | TBD | TBD |
| Max Logit | TBD | TBD | TBD | TBD |
| Energy | TBD | TBD | TBD | TBD |
| Mahalanobis | TBD | TBD | TBD | TBD |
| ViM | TBD | TBD | TBD | TBD |
| NECO | TBD | TBD | TBD | TBD |

### Neural Collapse Metrics
| Metric | Value |
|--------|-------|
| NC1 (Within-class variance) | TBD |
| NC2 (Between-class angle) | TBD |
| NC3 (Weight-mean alignment) | TBD |
| NC4 (Self-duality) | TBD |

<!-- 
## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tqdm>=4.65.0
PyYAML>=6.0
```
<>
-->

## References

- Original Neural Collapse papers
- OOD detection benchmarks
- ResNet architecture
