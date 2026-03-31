#include "../norm.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

// One thread per (batch, time) position; each thread normalizes over n_embd.
__global__ void norm_forward_kernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int batch_size,
    int sequence_length,
    int n_embd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_rows = batch_size * sequence_length;
    if (idx >= num_rows) {
        return;
    }

    const int batch_idx = idx / sequence_length;
    const int sequence_idx = idx % sequence_length;

    const int row_offset =
        batch_idx * sequence_length * n_embd + sequence_idx * n_embd;

    float mean = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        mean += input[row_offset + i];
    }
    mean /= static_cast<float>(n_embd);

    float var = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        const float d = input[row_offset + i] - mean;
        var += d * d;
    }
    var /= static_cast<float>(n_embd);
    const float inv_std = 1.0f / sqrtf(var + 1e-5f);

    for (int i = 0; i < n_embd; i++) {
        const float value = input[row_offset + i];
        output[row_offset + i] =
            ((value - mean) * inv_std) * weights[i] + biases[i];
    }
}

LayerNorm::LayerNorm(const std::vector<int>& shape) : _shape(shape) {
    init_weights(shape, 0.0f, 1.0f);
}

void LayerNorm::init_weights(
    const std::vector<int>& shape,
    float mean,
    float stddev) {
    if (shape.empty()) {
        throw std::invalid_argument("LayerNorm::init_weights: shape is empty");
    }
    const int c = shape.back();
    _weights.resize({c});
    _biases.resize({c});

    // GPT-2 convention: gamma initialized to 1, beta to 0.
    // (mean/stddev args are ignored here to match the standard LayerNorm init.)
    thrust::host_vector<float> h_w(_weights.size(), 1.0f);
    thrust::host_vector<float> h_b(_biases.size(), 0.0f);

    thrust::copy(h_w.begin(), h_w.end(), _weights.storage().begin());
    thrust::copy(h_b.begin(), h_b.end(), _biases.storage().begin());
}

void LayerNorm::forward(const Tensor& input, Tensor& output) {
    const auto input_shape = input.shape();

    if (input_shape.size() != 3) {
        throw std::invalid_argument(
            "LayerNorm::forward: input must have 3 dimensions");
    }

    const int c = input_shape[2];
    if (c != _weights.shape(-1) || c != _biases.shape(-1)) {
        throw std::invalid_argument(
            "LayerNorm::forward: last dim must match normalized size (gamma/beta)");
    }

    output.resize(input_shape);

    const int block = 256;
    const int grid =
        (input_shape[0] * input_shape[1] + block - 1) / block;
    norm_forward_kernel<<<grid, block>>>(
        input.data_ptr(),
        _weights.data_ptr(),
        _biases.data_ptr(),
        output.data_ptr(),
        input_shape[0],
        input_shape[1],
        input_shape[2]);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("LayerNorm::forward kernel launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("LayerNorm::forward kernel execution failed: ") +
            cudaGetErrorString(sync_err));
    }
}

void LayerNorm::set_params(
    const float* gamma,
    size_t gamma_size,
    const float* beta,
    size_t beta_size) {
    if (gamma_size != _weights.size() || beta_size != _biases.size()) {
        throw std::invalid_argument("LayerNorm::set_params: gamma/beta size mismatch");
    }
    thrust::copy(gamma, gamma + gamma_size, _weights.storage().begin());
    thrust::copy(beta, beta + beta_size, _biases.storage().begin());
}
