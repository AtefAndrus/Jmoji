# Experiment: v3_lr1e-4_top100_20251212

## Overview
- **Dataset**: v3 (2187 samples)
- **Experiment Type**: lr1e-4_top100
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
- Train: 1749
- Validation: 219
- Test: 219
- Unique Emojis: 100

## Results
| Metric | Value |
|--------|-------|
| Average Jaccard | 0.0000 |
| Exact Match Rate | 0.0000 |
| Micro F1 | 0.0000 |
| Avg Precision | 0.0000 |
| Avg Recall | 0.0000 |
| Avg F1 | 0.0000 |

## Diversity Metrics
| Metric | Value |
|--------|-------|
| Non-Top5 Ratio | 1.0000 |
| Unique Emojis in Output | 179 |
| Top5 Count | 0 |
| Non-Top5 Count | 227 |

## Training Info
- Best Checkpoint: /content/Jmoji/outputs/models/checkpoint-880
- Final Eval Loss: 8.085823059082031

## Notes
<!-- 実験に関するメモをここに記載 -->

