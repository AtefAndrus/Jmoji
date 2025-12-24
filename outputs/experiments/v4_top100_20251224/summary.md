# Experiment: v4_top100_20251224

## Overview
- **Dataset**: v4 (4237 samples)
- **Experiment Type**: top100
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
- Train: 3389
- Validation: 424
- Test: 424
- Unique Emojis: 100

## Results
| Metric | Value |
|--------|-------|
| Average Jaccard | 0.1196 |
| Exact Match Rate | 0.0000 |
| Micro F1 | 0.1942 |
| Avg Precision | 0.2231 |
| Avg Recall | 0.1757 |
| Avg F1 | 0.1935 |

## Diversity Metrics
| Metric | Value |
|--------|-------|
| Non-Top5 Ratio | 0.2101 |
| Unique Emojis in Output | 34 |
| Top5 Count | 1639 |
| Non-Top5 Count | 436 |

## Training Info
- Best Checkpoint: /content/Jmoji/outputs/models/checkpoint-5300
- Final Eval Loss: 4.659751892089844

## Notes
<!-- 実験に関するメモをここに記載 -->

