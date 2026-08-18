# Factorial findings: model x build

Each row holds every other factor fixed. `p` is a two-sided Fisher exact
test on pooled pass/fail items; a difference is only real when p < 0.05.

## Model effect, MLX build held fixed (3.8 vs 3.6)

- qwen3.8:27b-mlx: **76.0%** (152/200), 32.1 tok/s
- qwen3.6:27b-mlx: **74.0%** (148/200), 13.8 tok/s
- difference: +2.0 points, p = 0.729 -> not distinguishable
- speed ratio: 2.33x

| benchmark | qwen3.8:27b-mlx | qwen3.6:27b-mlx | delta |
|---|---|---|---|
| humaneval_plus | 90% | 82% | +8 |
| mbpp_plus | 62% | 66% | -4 |

## Model effect, GGUF build held fixed (3.8 vs 3.6)

- qwen3.8:27b: **77.0%** (154/200), 8.7 tok/s
- qwen3.6:27b: **74.0%** (148/200), 12.3 tok/s
- difference: +3.0 points, p = 0.561 -> not distinguishable
- speed ratio: 0.71x

| benchmark | qwen3.8:27b | qwen3.6:27b | delta |
|---|---|---|---|
| humaneval_plus | 91% | 80% | +11 |
| mbpp_plus | 63% | 68% | -5 |

## Build effect on Qwen 3.8 (MLX/nvfp4 vs GGUF/Q4_K_M)

- qwen3.8:27b-mlx: **76.0%** (152/200), 32.1 tok/s
- qwen3.8:27b: **77.0%** (154/200), 8.7 tok/s
- difference: -1.0 points, p = 0.906 -> not distinguishable
- speed ratio: 3.69x

| benchmark | qwen3.8:27b-mlx | qwen3.8:27b | delta |
|---|---|---|---|
| humaneval_plus | 90% | 91% | -1 |
| mbpp_plus | 62% | 63% | -1 |

## Build effect on Qwen 3.6 (MLX/nvfp4 vs GGUF/Q4_K_M)

- qwen3.6:27b-mlx: **74.0%** (148/200), 13.8 tok/s
- qwen3.6:27b: **74.0%** (148/200), 12.3 tok/s
- difference: +0.0 points, p = 1.000 -> not distinguishable
- speed ratio: 1.12x

| benchmark | qwen3.6:27b-mlx | qwen3.6:27b | delta |
|---|---|---|---|
| humaneval_plus | 82% | 80% | +2 |
| mbpp_plus | 66% | 68% | -2 |
