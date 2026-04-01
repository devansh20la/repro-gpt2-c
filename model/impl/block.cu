#include "../block.h"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

// Elementwise add for residual: c = a + b, all [B, T, C] contiguous.
__global__ void add_kernel(const float* a, const float* b, float* c, int B, int T, int C) {
    // 1. idx = blockIdx.x * blockDim.x + threadIdx.x over B*T*C elements.
    // 2. Guard idx < B*T*C.
    // 3. c[idx] = a[idx] + b[idx].
}


Block::Block(int in_features, int scaling_factor, int n_heads)
    : _in_features(in_features),
      _scaling_factor(scaling_factor),
      _ln1({in_features}),
      _attn(in_features, n_heads),
      _ln2({in_features}),
      _mlp(in_features, scaling_factor) {}

void Block::forward(const Tensor& input, Tensor& output) {
    // The forward pass of block layer is very interesting, as you need to 
    // deal with the residual connections. 

    // 1. Perform checks on the input and output shapes
    // 2. Resize output tensor

    // 3. Pass the input through the layer norm layer
    // 4. Pass the output through the causal multi-head attention layer

    // 5. Add residual connection between the input and the output of the causal multi-head attention layer


    // 5. Pass the output through the layer norm layer
    // 6. Pass the output through the MLP layer
    
    // 7. Add the residual connection between the input and the output of the MLP layer.
}

void Block::init_weights(float mean, float std) {
    _ln1.init_weights({_in_features}, mean, std);
    _attn.init_weights(mean, std);
    _ln2.init_weights({_in_features}, mean, std);
    _mlp.init_weights(mean, std);
}

void Block::load_weights(
    const std::unordered_map<std::string, std::vector<float>>& tensors,
    const std::string& prefix) {
    const auto& ln1_w = tensors.at(prefix + "ln1.weight");
    const auto& ln1_b = tensors.at(prefix + "ln1.bias");
    const auto& ln2_w = tensors.at(prefix + "ln2.weight");
    const auto& ln2_b = tensors.at(prefix + "ln2.bias");

    _ln1.set_params(ln1_w.data(), ln1_w.size(), ln1_b.data(), ln1_b.size());
    _ln2.set_params(ln2_w.data(), ln2_w.size(), ln2_b.data(), ln2_b.size());
    _attn.load_weights(tensors, prefix);
    _mlp.load_weights(tensors, prefix);
}
