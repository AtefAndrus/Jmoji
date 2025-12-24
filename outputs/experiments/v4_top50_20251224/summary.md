# Experiment: v4_top50_20251224

## Overview
- **Dataset**: v4 (1337 samples)
- **Experiment Type**: top50
- **Date**: 20251224
- **GPU**: NVIDIA A100-SXM4-40GB

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | sonoisa/t5-base-japanese |
| Epochs | 50 |
| Batch Size | 16 |
| Learning Rate | 0.0003 |
| Label Smoothing | 0.1 |
| Early Stopping | 5 epochs |
| FP16 | False |

## Data Split
- Train: 1069
- Validation: 134
- Test: 134
- Unique Emojis: 50

## Results
| Metric | Value |
|--------|-------|
| Average Jaccard | 0.1652 |
| Exact Match Rate | 0.0000 |
| Micro F1 | 0.2686 |
| Avg Precision | 0.3132 |
| Avg Recall | 0.2342 |
| Avg F1 | 0.2629 |

## Diversity Metrics
| Metric | Value |
|--------|-------|
| Non-Top5 Ratio | 0.2071 |
| Unique Emojis in Output | 16 |
| Top5 Count | 513 |
| Non-Top5 Count | 134 |

## Training Info
- Best Checkpoint: /content/Jmoji/outputs/models/checkpoint-1876
- Final Eval Loss: 4.195186614990234

## Notes
<!-- 実験に関するメモをここに記載 -->

