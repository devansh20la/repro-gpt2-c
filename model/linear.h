#pragma once

#include <cstddef>
#include <thrust/device_vector.h>
#include "../tensor.h"

class LinearLayer {
private:
    int _in_features;
    int _out_features;
    bool _bias;
    thrust::device_vector<float> _weights;  // [out_features, in_features]
    thrust::device_vector<float> _biases;   // [out_features]

public:
    LinearLayer(int in_features, int out_features, bool bias=true);

    // input shape:  [batch_size, in_features]
    // output shape: [batch_size, out_features]
    // batch_size is read from input.shape(0) — no separate parameter needed.
    void forward(const Tensor& input, Tensor& output);

    void init_weights(float mean, float std);
    void set_weights(const float* weights, size_t size);
    void set_biases(const float* biases, size_t size);
};
