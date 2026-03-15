# Boundary Divergence: A Geometric Diagnostic of Cross-Model Disagreement

This repository contains the code and notebooks for the paper:

> **Boundary Divergence: A Geometric Diagnostic of Cross-Model Disagreement**
> Elizabeth J. Taylor, 2026

## Overview

This paper introduces **boundary divergence**, a gradient-based metric that measures how differently two independently trained neural classifiers position their decision boundaries around the same input. We show that cross-model disagreement concentrates disproportionately in high-divergence regions, that this relationship is monotonic across divergence quantiles, and that the model with the lower boundary score is correct 80–85% of the time when models disagree.

## Repository Structure

```
boundary-divergence/
├── README.md
├── experiment1_same_architecture.ipynb   # GPT-2, seeds 42 & 99, SST-2 → IMDB/Amazon/Tweets
├── experiment2_five_seeds.ipynb          # GPT-2, 5 seeds, 10 pairwise combinations
├── experiment3_cross_architecture.ipynb  # DistilBERT vs BERT vs RoBERTa
└── experiment4_nli.ipynb                 # BERT vs RoBERTa, MNLI → SNLI
```

## Experiments

### Experiment 1 — Same Architecture, Different Seeds
- Models: GPT-2 (`seed=42`, `seed=99`)
- Training data: SST-2
- Evaluation: IMDB, Amazon Polarity, TweetEval (500 sentences each)
- Key result: High-divergence inputs are 2.0–2.6× more likely to produce cross-model disagreement

### Experiment 2 — Five Seeds, All Pairwise Combinations
- Models: GPT-2 (`seeds=42, 99, 123, 456, 789`)
- 10 pairwise combinations × 3 OOD datasets = 30 total tests
- Key result: 12/30 combinations significant; not all seed pairs produce geometrically divergent models

### Experiment 3 — Cross-Architecture
- Models: DistilBERT, BERT, RoBERTa trained on SST-2
- 9/9 architecture pairs significant
- Key result: Effect generalizes across model families, strongest for RoBERTa pairs

### Experiment 4 — NLI Generalization
- Models: BERT and RoBERTa trained on MNLI (50k samples)
- Evaluation: SNLI
- Key result: Asymmetry ratio 1.89× (95% CI: 1.34–2.71), Spearman r=0.182

## Requirements

```
torch
transformers
datasets
numpy
scipy
matplotlib
```

Install with:
```bash
pip install torch transformers datasets numpy scipy matplotlib
```

## Reproducing Results

Each notebook is self-contained and can be run end-to-end. Training takes approximately 30–60 minutes per notebook on a single GPU.

> **Note:** Output paths use `/kaggle/working/` by default. If running locally, change `output_dir` in the `TrainingArguments` block to a local path such as `./outputs/`.

## Key Metric

**Boundary score** for input $x$:

$$S(x) = \|\nabla_x \, m(x)\|_2$$

where $m(x) = f_{\text{pos}}(x) - f_{\text{neg}}(x)$ is the classification margin.

**Boundary divergence** between models $A$ and $B$:

$$D(x) = |S_A(x) - S_B(x)|$$

## Reproducibility Note

All experiments were verified to run end-to-end and reproduce the core findings. However, exact numerical results may differ slightly from those reported in the paper due to non-deterministic elements in model training (random weight initialization, GPU parallelism). The qualitative findings are stable across runs:

- Boundary divergence predicts cross-model disagreement significantly above chance
- The asymmetry effect is consistent in direction across all dataset-pair combinations
- The monotonic relationship between divergence and disagreement holds across runs

For Experiment 4 (NLI task), one verification run produced Spearman r=0.118 (p=0.008) and asymmetry ratio 1.43x, compared to r=0.182 (p<0.001) and 1.89x reported in the paper. Both runs are statistically significant and support the same conclusion.

To reduce variance, set explicit seeds in `TrainingArguments` and use `CUBLAS_WORKSPACE_CONFIG=:4096:8` before running on GPU.

## License

MIT License
