#include "../linear.h"

#include <random>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

__global__ void linear_forward_kernel(const float* input,
                                      const float* weights,
                                      const float* biases,
                                      float* output,
                                      int in_features,
                                      int out_features) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;  // output index
    if (o >= out_features) return;

    float acc = biases[o];
    for (int i = 0; i < in_features; ++i) {
        acc += input[i] * weights[o * in_features + i];
    }
    output[o] = acc;
}

LinearLayer::LinearLayer(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features) {
    _weights.resize(static_cast<size_t>(in_features_) * static_cast<size_t>(out_features_));
    _biases.resize(static_cast<size_t>(out_features_));
}

void LinearLayer::forward(const thrust::device_vector<float>& input,
                          thrust::device_vector<float>& output) {
    if (input.size() != static_cast<size_t>(in_features_)) {
        throw std::runtime_error("LinearLayer::forward: input size mismatch");
    }
    output.resize(static_cast<size_t>(out_features_));

    const int block = 256;
    const int grid = (out_features_ + block - 1) / block;

    linear_forward_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(input.data()),
        thrust::raw_pointer_cast(_weights.data()),
        thrust::raw_pointer_cast(_biases.data()),
        thrust::raw_pointer_cast(output.data()),
        in_features_,
        out_features_);
    cudaDeviceSynchronize();
}

void LinearLayer::init_weights() {
    thrust::host_vector<float> h_w(_weights.size());
    thrust::host_vector<float> h_b(_biases.size());

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(mean_, std_);

    for (size_t i = 0; i < h_w.size(); ++i) h_w[i] = dist(rng);
    for (size_t i = 0; i < h_b.size(); ++i) h_b[i] = 0.0f;

    thrust::copy(h_w.begin(), h_w.end(), _weights.begin());
    thrust::copy(h_b.begin(), h_b.end(), _biases.begin());
}
