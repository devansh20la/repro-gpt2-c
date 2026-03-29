#pragma once

#include "../tensor.h"

class LayerNorm {
private:
    std::vector<int> _shape;
    Tensor _weights;
    Tensor _biases;

public:
    LayerNorm(const std::vector<int>& shape);
    void init_weights(const std::vector<int>& shape, float mean, float stddev);
    void forward(const Tensor& input, Tensor& output);
};
