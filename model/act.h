#pragma once

#include "../tensor.h"

class ReLU {
public:
    ReLU() = default;

    // output[i] = max(input[i], 0)
    // Shape is preserved: output.shape() == input.shape()
    void forward(const Tensor& input, Tensor& output);
};


class Softmax {
public:
    Softmax() = default;

    // output[i] = exp(input[i]) / sum(exp(input)) along the last dimension
    void forward(const Tensor& input, Tensor& output);
};
