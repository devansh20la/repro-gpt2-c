# Reproduce GPT-2 in C++/CUDA

## Introduction
This repository serves as a guide to learn about writing cuda kernels for various layers 
in GPT2 implementations. The repository features a set of empty functions with comments 
to indicate "what?" to implement making it easier to learn. Some basic codes are filled
out for ease of understanding and "fill in the empty blanks" style of learning. 

You get **scaffolded code**: APIs and forward paths are wired up; kernels and tricky pieces are marked with comments so you know *what* to implement. Fill in the blanks, run, profile, and iterate.

## Who this is for

You’ll move fastest if you already have:

1. **Python & PyTorch** — comfortable training or debugging small models.
2. **How transformers work** — attention, residuals, layer norm at the level of shapes and data flow (e.g. [nanoGPT](https://github.com/karpathy/nanoGPT) is a good mental model).
3. **C++ basics** — enough to write arithmetic and follow host/device code; reading existing CUDA-style code matters more than memorizing the standard library.
4. **An NVIDIA GPU** on the machine where you build and run.

## Setup

1. **System toolchain and Python bits**

    ```bash
    sudo apt update && sudo apt -y upgrade && \
    sudo apt-get install -y python3 pip && \
    pip install requests numpy tiktoken && \
    sudo apt-get install -y g++-10
    ```

2. **CUDA Toolkit 13** (adjust the repo line if you use a different Ubuntu release)

    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
    sudo apt-get update && \
    sudo apt-get install -y cuda-13
    ```

3. **Pretrained Weights** — Download and convert Hugging Face checkpoints for this codebase using `utils/process_hf_weights.py`.

## Track your progress

Tick layers as you implement and verify them. The PyTorch column is your spec: same math, your kernels.

| Fwd | Layer | PyTorch Equivalent |
|---|---|---|
| [X] | `Embedding` | `nn.Embedding` |
| [X] | `LinearLayer` | `nn.Linear` |
| [X] | `ReLU` | `nn.ReLU` |
| [X] | `Softmax` | `nn.Softmax` |
| [X] | `Attention` | `F.scaled_dot_product_attention` |
| [X] | `CausalMultiheadedAttention` | Full attention block |
| [X] | `LayerNorm` | `nn.LayerNorm` |
| [X] | `GELU` | `nn.GELU` |
| [X] | `MLP` | `Linear -> GELU -> Linear` |
| [X] | `TransformerBlock` | `LN -> Attn -> res -> LN -> MLP -> res` |
| [X] | `GPT2` | Full model |
| [X] | `CrossEntropyLoss` | `nn.CrossEntropyLoss` |

## Where to start

Suggested order:
1. **Parallel programming context** — [CUDA lecture playlist](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr)) is perfect introduction to thread blocks and memory.
2. **Read the first three chapters** of *Programming Massively Parallel Processors* (through the core CUDA ideas)—enough to recognize launch configs and synchronization.
3. **Learn the layout of this repo** (then open files in this order):

    ```
    repro-gpt2-c/
    ├── main.cu                    # Entry point
    ├── gpt2_config.h              # GPT-2 hyperparameters (dims, heads, vocab, etc.)
    ├── tensor.h                   # Tensor helpers / device buffers
    ├── model.h / model.cu         # GPT-2 wiring; how layers compose
    ├── model/                     # One header per layer + shared impl folder
    │   ├── act.h
    │   ├── block.h
    │   ├── causal_attn.h
    │   ├── embedding.h
    │   ├── linear.h
    │   ├── mlp.h
    │   ├── norm.h
    │   └── impl/
    │       ├── act.cu
    │       ├── block.cu
    │       ├── causal_attn.cu
    │       ├── embedding.cu
    │       ├── linear.cu
    │       ├── mlp.cu
    │       └── norm.cu
    ├── data/
    │   ├── preprocess.py          # Tokenize / pack shards in .bin format
    │   ├── input.txt              # (example raw text)
    ├── utils/
    │   └── process_hf_weights.py  # Hugging Face weights → weights.bin for this code
    ```

    - **`model/*.h`** — public layer interfaces (what goes in and out).
    - **`model/impl/*.cu`** — CUDA kernels and forward implementations.
    - **`main.cu`** — dataset, loop, and glue code.
    - **`data/`** and **`utils/`** — Python for data prep and checkpoint conversion.

4. **Data pipeline** — use `data/preprocess.py` to build a binary shard, then implement loading in `main.cu` so you can feed real batches.

5. **`tensor.h`** — Implement basic tensor helpers you’ll call from every layer (shapes, raw pointers, splits, etc.).

6. **`model/`** — Implement layers one at a time; use the activation examples as a template for launches, error checks, and comments.

## Build and run

```bash
nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -o main main.cu model.cu --run
```

## Time per layer of GPT 2 model
You can always squeeze out more juice from the GPU with clever ways of tiling and writing kernels
Try to measure time for forward pass of each layer and see how far can you get. 

1. Add NVTX tags in `model.cu` (and optionally in `model/impl/block.cu` / submodules for block internals).
2. Use `nsys` to measure time. (See the youtube course to understand correct methodology to measure time)

*RTX 3060, input `[B,T] = [4,1024]`. **Avg (ms)** per `GPT2::forward` from `sys.log` (`nsys stats --report nvtx_sum --timeunit msec`, 12 instances).*

### End-to-end

| | Fwd (ms) |
|---|---|
| **GPT-2 forward (total)** | 18046.4112 |

### Input & LM head (NVTX in `model.cu`)

| Stage | Fwd (ms) | NVTX range |
|---|---|---|
| Token embedding (WTE) | 0.4649 | `GPT2/embedding_token` |
| Position embedding (WPE) | 0.3328 | `GPT2/embedding_position` |
| Add token + position embeddings | 0.1218 | `GPT2/add_token_and_position` |
| Each Tranformer Block | 770 | `GPT2/transformer_block/0` |
| Final LayerNorm (before logits) | 0.6302 | `GPT2/layer_norm_final` |
| Final Linear (linear to vocab; weights tied to WTE) | 8889.5739 | `GPT2/lm_head_linear` |
