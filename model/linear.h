#pragma once

#include <thrust/device_vector.h>

class LinearLayer {
private:
    int in_features_;
    int out_features_;
    thrust::device_vector<float> weights_;  // [out_features_, in_features_]
    thrust::device_vector<float> biases_;   // [out_features_]
    float mean_ = 0.0f;
    float std_ = 1.0f;

public:
    LinearLayer(int in_features, int out_features);

    // input size must be in_features_, output size must be out_features_.
    void forward(const thrust::device_vector<float>& input,
                 thrust::device_vector<float>& output);

    void init_weights();
};
