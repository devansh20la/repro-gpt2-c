#include "../causal_attn.h"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

__global__ void attn_scores_kernel(
    const float* query, 
    const float* key, 
    float* output,
    int B, int m, int n, int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * m * k) {
        const int batch_idx = idx / (m * k);
        const int local_idx = idx % (m * k);
        const int row = local_idx / k;
        const int col = local_idx % k;
        
        float res = 0.0f;
        for (int i = 0; i < n; i++) {
            res += query[batch_idx * m * n + row * n + i] * key[batch_idx * n * k + col * n + i];
        }

        if (row < col) {
            output[idx] = -INFINITY;
        } else {
            output[idx] = res / sqrtf(static_cast<float>(n));
        }
    }
}

__global__ void matmul_kernel(
    const float* matA,
    const float* matB,
    float* matC,
    int B, int m, int n, int k) {
    // matA shape is [B, m, n] or [B, T, T]
    // matB shape is [B, n, k] or [B, T, C]
    // matC shape is [B, m, k] or [B, T, C]

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < B * m * k) {
        const int batch_idx = idx / (m * k);
        const int local_idx = idx % (m * k);
        const int row = local_idx / k;
        const int col = local_idx % k;

        float res = 0.0f;
        for (int i = 0; i < n; i++) {
            res += matA[batch_idx * m * n + row * n + i] * matB[batch_idx * n * k + i * k + col];
        }

        matC[batch_idx * m * k + row * k + col] = res;
    }
}

CausalAttention::CausalAttention(int n_embd)
    : _n_embd(n_embd), 
      _proj(n_embd, 3 * n_embd), 
      _attention(n_embd) {
        init_weights(0.0f, 1.0f);
      }

void CausalAttention::forward(const Tensor& input, Tensor& output) {
    // input shape is [batch_size, seq_len, n_embd]
    // output shape is [batch_size, seq_len, n_embd]

    // project input -> query_key_value using LinearLayer
    Tensor query_key_value;
    query_key_value.resize({input.shape(0), input.shape(1), 3 * input.shape(2)});
    _proj.forward(input, query_key_value);
    
    // Split into Q, K, V
    std::vector<Tensor> query_key_value_parts = query_key_value.split(3, 2);
    Tensor query = query_key_value_parts[0];
    Tensor key = query_key_value_parts[1];
    Tensor value = query_key_value_parts[2];

    // apply attention to query, key, and value
    // and masking.
    // softmax(query @ key^T) * value / sqrt(d_k)
    // output shape is [batch_size, seq_len, n_embd]
    _attention.forward(query, key, value, output);
}

Attention::Attention(int n_embd)
    : _n_embd(n_embd), 
    _softmax() {}

void Attention::forward(
    const Tensor& query, 
    const Tensor& key, 
    const Tensor& value,
    Tensor& output) {
    // query_key_value shape is [batch_size, seq_len, 3 * n_embd]
    // output shape is [batch_size, seq_len, n_embd]
    
    // resize attention scores tensor and output tensor
    _attention_scores.resize({query.shape(0), query.shape(1), key.shape(1)});
    output.resize({query.shape(0), query.shape(1), value.shape(2)});

    int block = 256;
    int grid = (_attention_scores.size() + block - 1) / block;
    attn_scores_kernel<<<grid, block>>>(
        query.data_ptr(),
        key.data_ptr(),
        _attention_scores.data_ptr(),
        query.shape(0),
        query.shape(1),
        key.shape(2),
        query.shape(1)
    );
    
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Attention::forward kernel launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Attention::forward kernel execution failed: ") +
            cudaGetErrorString(sync_err));
    }
    

    // apply softmax to attention scores
    _softmax.forward(_attention_scores, _attention_scores);

    // apply attention scores to value
    grid = (output.size() + block - 1) / block;

    matmul_kernel<<<grid, block>>>(
        _attention_scores.data_ptr(),
        value.data_ptr(),
        output.data_ptr(),
        output.shape(0),
        output.shape(1),
        output.shape(1),
        value.shape(2)
    );

    launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Attention::forward kernel launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Attention::forward kernel execution failed: ") +
            cudaGetErrorString(sync_err));
    }
}

void CausalAttention::init_weights(float mean, float std) {
    _proj.init_weights(mean, std);
}