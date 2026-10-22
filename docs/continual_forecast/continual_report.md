# Continual Learning & Catastrophic Forgetting Analysis

## Cross-Stage Perplexity & Retention Matrix

### Strategy: **replay_and_ewc**

| Stage Completed | python_technical | systems_cuda | math_algorithms |
| --- | --- | --- | --- |
| After Stage 1 | PPL: 342.10 (Loss: 5.835) | PPL: 1608.83 (Loss: 7.383) | PPL: 1240.39 (Loss: 7.123) |
| After Stage 2 | PPL: 160.94 (Loss: 5.081) | PPL: 247.68 (Loss: 5.512) | PPL: 586.16 (Loss: 6.374) |
| After Stage 3 | PPL: 68.55 (Loss: 4.228) | PPL: 202.07 (Loss: 5.309) | PPL: 157.37 (Loss: 5.059) |

- **Max Stage Forgetting (PPL Delta)**: `0.00`
