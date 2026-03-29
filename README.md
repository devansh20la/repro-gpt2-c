# repro-gpt2-c

## Update python packages and GCC
```
sudo apt update && sudo apt -y upgrade && \
sudo apt-get install -y python3 pip && \
pip install requests numpy tiktoken && \
sudo apt-get install -y g++-10
```

## Update nvcc compiler
ensure your driver support cuda 13 and then run

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
sudo apt-get update && \
sudo apt-get install -y cuda-toolkit-13-0
```

## Build

```bash
nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -o train_gpt2 train_gpt2.cu model.cu 
  
```


# Layers
| Fwd | Bwd | Layer | PyTorch Equivalent |
|---|---|---|---|
| [x] | [ ] | `Embedding` | `nn.Embedding` |
| [x] | [ ] | `LinearLayer` | `nn.Linear` |
| [x] | [ ] | `ReLU` | `nn.ReLU` |
| [x] | [ ] | `Softmax` | `nn.Softmax` |
| [x] | [ ] | `Attention` | `F.scaled_dot_product_attention` |
| [x] | [ ] | `CausalMultiheadedAttention` | Full attention block |
| [x] | [ ] | `LayerNorm` | `nn.LayerNorm` |
| [x] | [ ] | `GELU` | `nn.GELU` |
| [x] | [ ] | `MLP` | `Linear -> GELU -> Linear` |
| [x] | [ ] | `TransformerBlock` | `LN -> Attn -> res -> LN -> MLP -> res` |
| [ ] | [ ] | `GPT2` | Full model |
| [ ] | [ ] | `CrossEntropyLoss` | `nn.CrossEntropyLoss` |
