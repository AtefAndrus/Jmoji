# Experiment: v4_focal_top50_20251224

## Overview
- **Dataset**: v4 (1337 samples)
- **Experiment Type**: focal_top50
- **Date**: 20251224
- **GPU**: NVIDIA A100-SXM4-40GB

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | sonoisa/t5-base-japanese |
| Epochs | 50 |
| Batch Size | 16 |
| Learning Rate | 0.0003 |
| Label Smoothing | 0.0 |
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
| Average Jaccard | 0.1816 |
| Exact Match Rate | 0.0000 |
| Micro F1 | 0.2863 |
| Avg Precision | 0.3448 |
| Avg Recall | 0.2501 |
| Avg F1 | 0.2841 |

## Diversity Metrics
| Metric | Value |
|--------|-------|
| Non-Top5 Ratio | 0.1391 |
| Unique Emojis in Output | 16 |
| Top5 Count | 563 |
| Non-Top5 Count | 91 |

## Training Info
- Best Checkpoint: /content/Jmoji/outputs/models/checkpoint-1876
- Final Eval Loss: 2.9487485885620117

## Notes
<!-- 実験に関するメモをここに記載 -->

