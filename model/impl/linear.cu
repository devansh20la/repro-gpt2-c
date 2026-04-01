#include "../linear.h"

#include <random>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

// One thread per (flattened batch position, output feature) pair — or choose your own tiling.
// weights: [out_features, in_features] row-major: weight[out_j, in_i] at weights[out_j * in_features + in_i].
__global__ void linear_forward_kernel() {
    // 1. Map thread index -> (batch_idx, out_j) covering batch_size * out_feature_size outputs.
    // 2. Bounds-check.
    // 3. Initialize accumulator with biases[out_j] if bias, else 0.
    // 4. Dot row batch_idx of input with column out_j of W: sum_i input[...] * weights[out_j * in + i].
    // 5. Write output[batch_idx, out_j].
}

static void check_cuda(const char* what, cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

LinearLayer::LinearLayer(int in_features, int out_features, bool bias)
    : _in_features(in_features), 
    _out_features(out_features), 
    _bias(bias) {
        // 1. resize the weights and biases of the linear layer
}

void LinearLayer::forward(const Tensor& input, Tensor& output) {
    // 1. Perform checks on the input and output shapes
    // 2. Resize output tensor
    // 3. Launch the linear forward kernel
    // 4. Perform checks for errors in the kernel launch and execution
    // 5. Synchronize the device to ensure the kernel has completed
}

void LinearLayer::init_weights(float mean, float std) {
    // 1. Initialize the weights of the linear layer
    // 2. Initialize the biases of the linear layer
}

void LinearLayer::set_weights(const float* weights, size_t size) {
    // 1. Set the weights of the linear layer
}

void LinearLayer::set_biases(const float* biases, size_t size) {
    // 1. Set the biases of the linear layer
}
