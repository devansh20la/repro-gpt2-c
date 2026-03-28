#pragma once

#include "../tensor.h"

class ReLU {
public:
    ReLU() = default;

    // output[i] = max(input[i], 0)
    // Shape is preserved: output.shape() == input.shape()
    void forward(const Tensor& input, Tensor& output);
};


// GELU — Gaussian Error Linear Unit (tanh approximation)
//
// GPT-2 uses this instead of ReLU. Where ReLU has a hard cutoff at 0,
// GELU is a smooth curve that gradually turns off for negative values:
//
//   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// PyTorch equivalent: nn.GELU(approximate='tanh')
class GELU {
public:
    GELU() = default;

    // output[i] = gelu(input[i])
    // Shape is preserved: output.shape() == input.shape()
    void forward(const Tensor& input, Tensor& output);
};


class Softmax {
public:
    Softmax() = default;

    // output[i] = exp(input[i]) / sum(exp(input)) along the last dimension
    void forward(const Tensor& input, Tensor& output);
};
