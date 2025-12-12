# Experiment: v3_top100_20251212

## Overview
- **Dataset**: v3 (2187 samples)
- **Experiment Type**: top100
- **Date**: 20251212
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
- Train: 1749
- Validation: 219
- Test: 219
- Unique Emojis: 100

## Results
| Metric | Value |
|--------|-------|
| Average Jaccard | 0.0581 |
| Exact Match Rate | 0.0000 |
| Micro F1 | 0.0939 |
| Avg Precision | 0.0974 |
| Avg Recall | 0.0898 |
| Avg F1 | 0.0910 |

## Diversity Metrics
| Metric | Value |
|--------|-------|
| Non-Top5 Ratio | 0.1978 |
| Unique Emojis in Output | 20 |
| Top5 Count | 442 |
| Non-Top5 Count | 109 |

## Training Info
- Best Checkpoint: /content/Jmoji/outputs/models/checkpoint-2640
- Final Eval Loss: 4.714391231536865

## Notes
<!-- 実験に関するメモをここに記載 -->

