# Continual Learning & Catastrophic Forgetting Analysis

## Cross-Stage Perplexity & Retention Matrix

### Strategy: **naive**

| Stage Completed | python_technical | systems_cuda | math_algorithms |
| --- | --- | --- | --- |
| After Stage 1 | PPL: 36.06 (Loss: 3.585) | PPL: 880.92 (Loss: 6.781) | PPL: 454.08 (Loss: 6.118) |
| After Stage 2 | PPL: 90.17 (Loss: 4.502) | PPL: 32.53 (Loss: 3.482) | PPL: 473.10 (Loss: 6.159) |
| After Stage 3 | PPL: 71.78 (Loss: 4.274) | PPL: 72.40 (Loss: 4.282) | PPL: 21.62 (Loss: 3.074) |

- **Max Stage Forgetting (PPL Delta)**: `39.87`

### Strategy: **replay_15pct**

| Stage Completed | python_technical | systems_cuda | math_algorithms |
| --- | --- | --- | --- |
| After Stage 1 | PPL: 36.06 (Loss: 3.585) | PPL: 880.92 (Loss: 6.781) | PPL: 454.08 (Loss: 6.118) |
| After Stage 2 | PPL: 46.82 (Loss: 3.846) | PPL: 33.72 (Loss: 3.518) | PPL: 387.30 (Loss: 5.959) |
| After Stage 3 | PPL: 26.36 (Loss: 3.272) | PPL: 80.11 (Loss: 4.383) | PPL: 25.17 (Loss: 3.226) |

- **Max Stage Forgetting (PPL Delta)**: `46.39`

### Strategy: **ewc_backbone**

| Stage Completed | python_technical | systems_cuda | math_algorithms |
| --- | --- | --- | --- |
| After Stage 1 | PPL: 36.06 (Loss: 3.585) | PPL: 880.92 (Loss: 6.781) | PPL: 454.08 (Loss: 6.118) |
| After Stage 2 | PPL: 79.26 (Loss: 4.373) | PPL: 30.73 (Loss: 3.425) | PPL: 507.81 (Loss: 6.230) |
| After Stage 3 | PPL: 87.48 (Loss: 4.471) | PPL: 92.12 (Loss: 4.523) | PPL: 25.99 (Loss: 3.258) |

- **Max Stage Forgetting (PPL Delta)**: `61.39`

### Strategy: **ewc_forecast**

| Stage Completed | python_technical | systems_cuda | math_algorithms |
| --- | --- | --- | --- |
| After Stage 1 | PPL: 36.06 (Loss: 3.585) | PPL: 880.92 (Loss: 6.781) | PPL: 454.08 (Loss: 6.118) |
| After Stage 2 | PPL: 78.40 (Loss: 4.362) | PPL: 30.97 (Loss: 3.433) | PPL: 501.20 (Loss: 6.217) |
| After Stage 3 | PPL: 83.26 (Loss: 4.422) | PPL: 88.34 (Loss: 4.481) | PPL: 24.48 (Loss: 3.198) |

- **Max Stage Forgetting (PPL Delta)**: `57.37`

### Strategy: **replay_and_ewc**

| Stage Completed | python_technical | systems_cuda | math_algorithms |
| --- | --- | --- | --- |
| After Stage 1 | PPL: 36.06 (Loss: 3.585) | PPL: 880.92 (Loss: 6.781) | PPL: 454.08 (Loss: 6.118) |
| After Stage 2 | PPL: 40.84 (Loss: 3.710) | PPL: 33.15 (Loss: 3.501) | PPL: 374.41 (Loss: 5.925) |
| After Stage 3 | PPL: 26.27 (Loss: 3.269) | PPL: 76.67 (Loss: 4.339) | PPL: 25.81 (Loss: 3.251) |

- **Max Stage Forgetting (PPL Delta)**: `43.52`
