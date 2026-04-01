#pragma once

#include <cstddef>
#include <thrust/device_vector.h>
#include "../tensor.h"

// PyTorch: `nn.Linear(in_features, out_features, bias=True)`
// Computes: output = input @ W^T + b  with W shape [out_features, in_features] (row-major on device).
class LinearLayer {
private:
    int _in_features;
    int _out_features;
    bool _bias;
    thrust::device_vector<float> _weights;  // [out_features, in_features] row-major
    thrust::device_vector<float> _biases;   // [out_features] (if bias)

public:
    LinearLayer(int in_features, int out_features, bool bias = true);

    // input shape:  [batch_size, in_features]
    // forward pass the input through the linear layer
    void forward(const Tensor& input, Tensor& output);

    // initialize the weights of the linear layer
    void init_weights(float mean, float std);

    // set the weights of the linear layer
    void set_weights(const float* weights, size_t size);

    // set the biases of the linear layer
    void set_biases(const float* biases, size_t size);
};
