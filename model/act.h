#pragma once

#include <thrust/device_vector.h>

class ReLU {
public:
    ReLU() = default;

    // output[i] = max(input[i], 0)
    void forward(const thrust::device_vector<float>& input,
                 thrust::device_vector<float>& output);
};
