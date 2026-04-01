#include "../mlp.h"

#include <cuda_runtime.h>

// No custom __global__ kernels here — composition of Linear + GELU.
// Shapes: [B, T, in] -> fc1 -> [B, T, in*scaling] -> GELU -> fc2 -> [B, T, in].

MLP::MLP(int in_features, int scaling_factor)
    : _in_features(in_features),
      _scaling_factor(scaling_factor),
      _fc1(in_features, in_features * scaling_factor),
      _gelu(),
      _fc2(in_features * scaling_factor, in_features) {}

void MLP::forward(const Tensor& input, Tensor& output) {
    // 1. Allocate hidden activations: shape [B, T, in_features * scaling_factor].
    // 2. _fc1.forward(input, hidden).
    // 3. _gelu.forward(hidden, hidden)  (in-place on the same buffer is fine if shapes match).
    // 4. _fc2.forward(hidden, output).

    // 1. Allocate hidden activations: shape [B, T, in_features * scaling_factor].
    Tensor _fc1_output({input.shape(0), input.shape(1), _in_features * _scaling_factor});
    _fc1.forward(input, _fc1_output);

    _gelu.forward(_fc1_output, _fc1_output);

    _fc2.forward(_fc1_output, output);
}

void MLP::init_weights(float mean, float std) {
    _fc1.init_weights(mean, std);
    _fc2.init_weights(mean, std);
}

void MLP::load_weights(
    const std::unordered_map<std::string, std::vector<float>>& tensors,
    const std::string& prefix) {
    // This function takes a map of strings to vectors of floats and a prefix.
    // and loads the weights of the MLP layer from the file.
    // You are only calling the load_weights function of the linear layers with the appropriate weights and biases.
}
