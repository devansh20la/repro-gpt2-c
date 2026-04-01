#include "../norm.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

// Typical layout: input [B, T, C]; one thread per (b, t) row, normalize over C.
// Note that layer norm behaves differently in training and inference.
// In training, we compute the mean and variance of the input and use them to normalize the input.
// In inference, we use the running mean and variance computed during training.
// We will assume that we are in inference mode.
__global__ void norm_forward_kernel() {
    // 1. Map thread -> linear index over rows: num_rows = batch_size * sequence_length.
    // 2. Decode row -> (batch_idx, seq_idx); compute row_offset into flattened [B,T,C] buffer.
    // 3. First pass: mean over C; second pass: variance; inv_std = rsqrt(var + eps).
    // 4. For each i in 0..n_embd-1: out = (in - mean) * inv_std * gamma[i] + beta[i].
}


LayerNorm::LayerNorm(const std::vector<int>& shape) : _shape(shape) {
    init_weights(shape, 0.0f, 1.0f);
}

void LayerNorm::init_params(
    const std::vector<int>& shape,
    float mean,
    float stddev) {
    (void)mean;
    (void)stddev;
    if (shape.empty()) {
        throw std::invalid_argument("LayerNorm::init_weights: shape is empty");
    }
    const int c = shape.back();
    _weights.resize({c});
    _biases.resize({c});

    thrust::host_vector<float> h_w(_weights.size(), 1.0f);
    thrust::host_vector<float> h_b(_biases.size(), 0.0f);

    thrust::copy(h_w.begin(), h_w.end(), _weights.storage().begin());
    thrust::copy(h_b.begin(), h_b.end(), _biases.storage().begin());
}

void LayerNorm::forward(const Tensor& input, Tensor& output) {
    // 1. Perform checks on the input and output shapes
    // 2. Resize output tensor
    // 3. Launch the layer norm forward kernel
    // 4. Perform checks for errors in the kernel launch and execution
    // 5. Synchronize the device to ensure the kernel has completed
}

void LayerNorm::set_params(
    const float* gamma,
    size_t gamma_size,
    const float* beta,
    size_t beta_size) {
    // 1. Perform checks on the gamma and beta sizes.
    // 2. Copy the parameters to the weights and biases. 
    if (gamma_size != _weights.size() || beta_size != _biases.size()) {
        throw std::invalid_argument("LayerNorm::set_params: gamma/beta size mismatch");
    }
    thrust::copy(gamma, gamma + gamma_size, _weights.storage().begin());
    thrust::copy(beta, beta + beta_size, _biases.storage().begin());
}
