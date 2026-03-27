# repro-gpt2-c

## Update OS
```
sudo apt update && sudo apt -y upgrade
sudo apt-get install -y python3 pip

pip install numpy, tiktoken
```
## Build

```bash
nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -o train_gpt2 \
  train_gpt2.cu \
  model/impl/linear.cu \
  model/impl/act.cu \
  model/impl/embedding.cu
```

# Layers
| Fwd | Bwd | Layer | PyTorch Equivalent |
|---|---|---|---|
| [x] | [ ] | `Embedding` | `nn.Embedding` |
| [x] | [ ] | `LinearLayer` | `nn.Linear` |
| [x] | [ ] | `ReLU` | `nn.ReLU` |
| [x] | [ ] | `Softmax` | `nn.Softmax` |
| [x] | [ ] | `Attention` | `F.scaled_dot_product_attention` |
| [x] | [ ] | `CausalAttention` | Full attention block |
| [ ] | [ ] | `LayerNorm` | `nn.LayerNorm` |
| [ ] | [ ] | `GELU` | `nn.GELU` |
| [ ] | [ ] | `MLP` | `Linear -> GELU -> Linear` |
| [ ] | [ ] | `TransformerBlock` | `LN -> Attn -> res -> LN -> MLP -> res` |
| [ ] | [ ] | `GPT2` | Full model |
| [ ] | [ ] | `CrossEntropyLoss` | `nn.CrossEntropyLoss` |
