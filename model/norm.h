#pragma once

#include "../tensor.h"

// PyTorch: `nn.LayerNorm(normalized_shape)` over the last dimension (GPT-2 applies on C for [B,T,C]).
// y = (x - mean) / sqrt(var + eps) * gamma + beta, per row over the last dim.
class LayerNorm {
private:
    std::vector<int> _shape;
    Tensor _weights;  // gamma (learned scale), shape [C]
    Tensor _biases;   // beta (learned shift), shape [C]

public:
    // Constructor to initialize the layer norm layer
    LayerNorm(const std::vector<int>& shape);

    // initialize the weights of the layer norm layer
    // Again, this isn't necessary, since we intend to load from pretrained weights, 
    // but it is a good practice to initialize the weights of the model.
    // for cases when we want to pass a random input to the model.
    void init_params(const std::vector<int>& shape, float mean, float stddev);

    // forward pass the input through the layer norm layer
    void forward(const Tensor& input, Tensor& output);

    // set the parameters of the layer norm layer
    // Different name cause its different than setting weights, but the functionality is the same.
    void set_params(const float* gamma, size_t gamma_size, const float* beta, size_t beta_size);
};
