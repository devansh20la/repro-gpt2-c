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