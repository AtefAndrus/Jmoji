# Experiment: v4_lr1e-4_20251223

## Overview
- **Dataset**: v4 (20000 samples)
- **Experiment Type**: lr1e-4
- **Date**: 20251223
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
- Train: 16000
- Validation: 2000
- Test: 2000
- Unique Emojis: 1300

## Results
| Metric | Value |
|--------|-------|
| Average Jaccard | 0.0661 |
| Exact Match Rate | 0.0000 |
| Micro F1 | 0.1115 |
| Avg Precision | 0.1366 |
| Avg Recall | 0.0970 |
| Avg F1 | 0.1113 |

## Diversity Metrics
| Metric | Value |
|--------|-------|
| Non-Top5 Ratio | 0.1919 |
| Unique Emojis in Output | 45 |
| Top5 Count | 7987 |
| Non-Top5 Count | 1897 |

## Training Info
- Best Checkpoint: /content/Jmoji/outputs/models/checkpoint-9000
- Final Eval Loss: 5.743080139160156

## Notes
<!-- 実験に関するメモをここに記載 -->
