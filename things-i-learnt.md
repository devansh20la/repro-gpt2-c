# Things I Learnt

## nvcc compilation flags

| Flag | What it does |
|---|---|
| `nvcc` | The NVIDIA CUDA compiler. It handles both GPU code (`__global__` kernels) and host C++ code. Under the hood, it splits your `.cu` file — GPU code goes to NVIDIA's PTX compiler, host code gets forwarded to a regular C++ compiler. |
| `-ccbin /usr/bin/g++-10` | **C++ compiler bin.** Tells nvcc which host compiler to use for the non-GPU parts. Without this, nvcc picks the system default `g++`. Some C++20 features need gcc-10+, so we pin it explicitly. |
| `-std=c++20` | **C++ standard.** Enables C++20 features like `std::span`, structured bindings, etc. This flag is passed through to `g++-10`. |
| `-O2` | **Optimization level 2.** Tells both the host compiler (g++) and the device compiler (ptxas) to optimize for speed — inlining, loop unrolling, register allocation, etc. Levels go from `-O0` (no optimization, fast compile, good for debugging) to `-O3` (aggressive). `-O2` is the standard production choice. |
| `-o train_gpt2` | **Output filename.** The compiled binary. Without this, nvcc defaults to `a.out`. |
| `train_gpt2.cu model/impl/*.cu` | **Source files.** All the `.cu` files to compile and link together. nvcc compiles each one separately, then links them into a single binary. |

### Compilation pipeline

```
train_gpt2.cu ──→ nvcc ──→ splits into:
                            ├─ GPU code ──→ ptxas ──→ device object
                            └─ CPU code ──→ g++-10 ──→ host object
                                                          │
model/impl/*.cu ──→ (same process for each)               │
                                                          ▼
                                                      linker ──→ train_gpt2 binary
```

## CUDA math functions: use the `f` suffix

GPU-optimized math functions for `float` (32-bit) have an `f` suffix. Without it, you get the `double` (64-bit) version, which is 2-32x slower on most GPUs.

| float (fast) | double (slow) | What it does |
|---|---|---|
| `expf(x)` | `exp(x)` | e^x |
| `fmaxf(a, b)` | `fmax(a, b)` | max of two values |
| `sqrtf(x)` | `sqrt(x)` | square root |
| `logf(x)` | `log(x)` | natural log |
| `fabsf(x)` | `fabs(x)` | absolute value |

Rule of thumb: always use `f`-suffixed functions when your data is `float`.

## Integer division (`/`) and modulo (`%`) for index decomposition

CUDA gives each thread a flat index (`idx = 0, 1, 2, ...`). Use `/` and `%` to decompose it into multi-dimensional coordinates.

- `/` strips away the lower dimensions
- `%` isolates them

Example: output shape `[B, m, k]`, total `B*m*k` threads:

```cpp
const int batch = idx / (m * k);   // which batch
const int local = idx % (m * k);   // position within batch
const int row   = local / k;       // which row
const int col   = local % k;       // which column
```

## `thrust::device_vector` vs raw pointers vs views

| Type | Owns memory? | Use for |
|---|---|---|
| `thrust::device_vector<T>` | Yes | Allocating and managing GPU buffers |
| `thrust::device_ptr<T>` | No | Passing to thrust algorithms (tagged as device memory) |
| `T*` (raw pointer) | No | Passing to CUDA `__global__` kernels |

`thrust::device_vector` has no "view" mode. For views (non-owning references to sub-ranges), use a raw pointer + shape.

## Causal masking

- Apply the mask **after** the full dot product, not inside the inner loop
- Use `-INFINITY` (not `0`) for masked positions, because `softmax(-inf) = 0` but `softmax(0) > 0`
- Condition: `col > row` means "future token" — block it

## Numerically stable softmax

Always subtract the max before exponentiating to avoid `exp` overflow:

```cpp
// Step 1: max for stability
float max_val = input[offset];
for (int i = 1; i < n; i++)
    max_val = fmaxf(max_val, input[offset + i]);

// Step 2: exp(x - max) and sum
float sum = 0.0f;
for (int i = 0; i < n; i++) {
    output[offset + i] = expf(input[offset + i] - max_val);
    sum += output[offset + i];
}

// Step 3: normalize
for (int i = 0; i < n; i++)
    output[offset + i] /= sum;
```

## `cudaMemcpy2D` for strided copies

Used for splitting tensors along non-contiguous dimensions (e.g. splitting `[B, T, 3C]` into three `[B, T, C]` chunks along the last dim).

```
cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)
```

- `spitch` = source row width in bytes (full row)
- `dpitch` = destination row width in bytes (chunk row)
- `width` = bytes to copy per row
- `height` = number of rows
