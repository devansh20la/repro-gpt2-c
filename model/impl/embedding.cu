#include "../embedding.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

// token ids: uint16_t; table: float[num_embeddings, out_feature_size].
__global__ void embedding_forward_kernel() {
    // 1. idx = blockIdx.x * blockDim.x + threadIdx.x; guard idx < total_tokens.
    // 2. Read token id t = input[idx].
    // 3. If t >= num_embeddings, optionally zero-fill or assert (your choice).
    // 4. Copy embeddings[t * out_feature_size + k] -> output[idx * out_feature_size + k] for all k.
}

static void check_cuda(const char* what, cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

Embedding::Embedding(int num_embeddings, int out_feature_size)
    : _num_embeddings(num_embeddings),
      _out_feature_size(out_feature_size) {
    _embeddings.resize(static_cast<size_t>(num_embeddings) * static_cast<size_t>(out_feature_size));
    init_embeddings(0.0f, 1.0f);
}

void Embedding::forward(const TensorBase<uint16_t>& input, Tensor& output) const {
    const int B = input.shape(0);
    const int N = input.shape(1);
    const int total_tokens = B * N;

    // 1. Perform checks on the input and output shapes
    // 2. Resize output tensor
    // 3. Launch the embedding forward kernel
    // 4. Perform checks for errors in the kernel launch and execution
    // 5. Synchronize the device to ensure the kernel has completed
}

void Embedding::init_embeddings(const float mean, const float std) {
    // 1. Initialize the embeddings of the embedding layer
    // If you write once initializer for any layer, this should be easy to write for other layers as well.
}

void Embedding::set_weights(const float* weights, size_t size) {
    // 1. Set the weights of the embedding layer
    // 
}
