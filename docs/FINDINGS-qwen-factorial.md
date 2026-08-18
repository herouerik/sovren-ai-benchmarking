# Factorial findings: model x build

Each row holds every other factor fixed. `p` is a two-sided Fisher exact
test on pooled pass/fail items; a difference is only real when p < 0.05.

## Model effect, MLX build held fixed (3.8 vs 3.6)

- qwen3.8:27b-mlx: **78.5%** (314/400), 31.3 tok/s
- qwen3.6:27b-mlx: **78.0%** (312/400), 13.6 tok/s
- difference: +0.5 points, p = 0.932 -> not distinguishable
- speed ratio: 2.30x

| benchmark | qwen3.8:27b-mlx | qwen3.6:27b-mlx | delta |
|---|---|---|---|
| humaneval | 93% | 85% | +8 |
| mbpp | 68% | 68% | +0 |
| spider | 72% | 72% | +0 |
| bfcl | 81% | 87% | -6 |

## Model effect, GGUF build held fixed (3.8 vs 3.6)

- qwen3.8:27b: **79.8%** (319/400), 8.9 tok/s
- qwen3.6:27b: **79.2%** (317/400), 12.4 tok/s
- difference: +0.5 points, p = 0.930 -> not distinguishable
- speed ratio: 0.72x

| benchmark | qwen3.8:27b | qwen3.6:27b | delta |
|---|---|---|---|
| humaneval | 96% | 83% | +13 |
| mbpp | 70% | 72% | -2 |
| spider | 72% | 75% | -3 |
| bfcl | 81% | 87% | -6 |

## Build effect on Qwen 3.8 (MLX/nvfp4 vs GGUF/Q4_K_M)

- qwen3.8:27b-mlx: **78.5%** (314/400), 31.3 tok/s
- qwen3.8:27b: **79.8%** (319/400), 8.9 tok/s
- difference: -1.2 points, p = 0.728 -> not distinguishable
- speed ratio: 3.50x

| benchmark | qwen3.8:27b-mlx | qwen3.8:27b | delta |
|---|---|---|---|
| humaneval | 93% | 96% | -3 |
| mbpp | 68% | 70% | -2 |
| spider | 72% | 72% | +0 |
| bfcl | 81% | 81% | +0 |

## Build effect on Qwen 3.6 (MLX/nvfp4 vs GGUF/Q4_K_M)

- qwen3.6:27b-mlx: **78.0%** (312/400), 13.6 tok/s
- qwen3.6:27b: **79.2%** (317/400), 12.4 tok/s
- difference: -1.2 points, p = 0.730 -> not distinguishable
- speed ratio: 1.10x

| benchmark | qwen3.6:27b-mlx | qwen3.6:27b | delta |
|---|---|---|---|
| humaneval | 85% | 83% | +2 |
| mbpp | 68% | 72% | -4 |
| spider | 72% | 75% | -3 |
| bfcl | 87% | 87% | +0 |
