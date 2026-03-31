#include "../mlp.h"

#include <cuda_runtime.h>

MLP::MLP(int in_features, int scaling_factor)
    : _in_features(in_features),
      _scaling_factor(scaling_factor),
      _fc1(in_features, in_features * scaling_factor),
      _gelu(),
      _fc2(in_features * scaling_factor, in_features) {}

void MLP::forward(const Tensor& input, Tensor& output) {
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
    // Expected keys (prefix already includes trailing dot):
    //   {prefix}mlp.c_fc.weight
    //   {prefix}mlp.c_fc.bias
    //   {prefix}mlp.c_proj.weight
    //   {prefix}mlp.c_proj.bias
    const auto& w_fc = tensors.at(prefix + "mlp.c_fc.weight");
    const auto& b_fc = tensors.at(prefix + "mlp.c_fc.bias");
    const auto& w_pr = tensors.at(prefix + "mlp.c_proj.weight");
    const auto& b_pr = tensors.at(prefix + "mlp.c_proj.bias");

    _fc1.set_weights(w_fc.data(), w_fc.size());
    _fc1.set_biases(b_fc.data(), b_fc.size());
    _fc2.set_weights(w_pr.data(), w_pr.size());
    _fc2.set_biases(b_pr.data(), b_pr.size());
}