# Experiment: v3_lr1e-4_20251212

## Overview
- **Dataset**: v3 (5000 samples)
- **Experiment Type**: lr1e-4
- **Date**: 20251212
- **GPU**: NVIDIA A100-SXM4-40GB

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | sonoisa/t5-base-japanese |
| Epochs | 50 |
| Batch Size | 16 |
| Learning Rate | 0.0001 |
| Label Smoothing | 0.1 |
| Early Stopping | 5 epochs |
| FP16 | False |

## Data Split
- Train: 4000
- Validation: 500
- Test: 500
- Unique Emojis: 785

## Results
| Metric | Value |
|--------|-------|
| Average Jaccard | 0.0481 |
| Exact Match Rate | 0.0000 |
| Micro F1 | 0.0805 |
| Avg Precision | 0.0853 |
| Avg Recall | 0.0700 |
| Avg F1 | 0.0753 |

## Diversity Metrics
| Metric | Value |
|--------|-------|
| Non-Top5 Ratio | 0.1087 |
| Unique Emojis in Output | 17 |
| Top5 Count | 1213 |
| Non-Top5 Count | 148 |

## Training Info
- Best Checkpoint: /content/Jmoji/outputs/models/checkpoint-11500
- Final Eval Loss: 5.648146629333496

## Notes
<!-- 実験に関するメモをここに記載 -->
