#include "../embedding.h"

// Implementation notes:
// - token ids are uint16_t (matches data/preprocess.py dtype)
// - embedding table is float[vocab_size, out_feature_size]

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

__global__ void embedding_forward_kernel(
    const uint16_t* input,
    const float* embeddings,
    float* output,
    int total_tokens,
    int num_embeddings,
    int out_feature_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_tokens) {
        const uint16_t input_idx = input[idx];

        if (input_idx >= num_embeddings) {
            for (int i = 0; i < out_feature_size; i++) {
                output[idx * out_feature_size + i] = 0.0f;
            }
            return;
        }

        for (int i = 0; i < out_feature_size; i++) {
            output[idx * out_feature_size + i] = embeddings[input_idx * out_feature_size + i];
        }
    }
}

Embedding::Embedding(
    int num_embeddings,
    int out_feature_size)
    : _num_embeddings(num_embeddings),
      _out_feature_size(out_feature_size) {

    _embeddings.resize(static_cast<size_t>(num_embeddings) * static_cast<size_t>(out_feature_size));
    
    init_embeddings(0.0f, 1.0f);
}

void Embedding::forward(
    const TensorBase<uint16_t>& input,
    Tensor& output) const {

    // input is [B, N] — read both dims from the tensor's shape
    const int B = input.shape(0);
    const int N = input.shape(1);
    const int total_tokens = B * N;

    // output is [B, N, out_feature_size]
    output.resize({B, N, _out_feature_size});

    const int num_threads = 256;
    const int num_blocks = (total_tokens + num_threads - 1) / num_threads;

    // The kernel treats the data as flat (total_tokens elements).
    // The [B, N] and [B, N, C] shapes are just metadata on the host side —
    // the GPU buffer is contiguous either way.
    embedding_forward_kernel<<<num_blocks, num_threads>>>(
        input.data_ptr(),
        thrust::raw_pointer_cast(_embeddings.data()),
        output.data_ptr(),
        total_tokens,
        _num_embeddings,
        _out_feature_size);

    cudaDeviceSynchronize();
}

void Embedding::init_embeddings(
    const float mean, 
    const float std) {
    thrust::host_vector<float> h_embeddings(_embeddings.size());

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(mean, std);

    for (size_t i = 0; i < h_embeddings.size(); ++i) h_embeddings[i] = dist(rng);

    thrust::copy(h_embeddings.begin(), h_embeddings.end(), _embeddings.begin());
}

void Embedding::set_weights(const float* weights, size_t size) {
    if (size != _embeddings.size()) {
        throw std::invalid_argument("Embedding::set_weights: size must equal embeddings.size()");
    }
    thrust::copy(weights, weights + size, _embeddings.begin());
}
