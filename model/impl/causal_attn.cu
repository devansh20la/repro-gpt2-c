#include "../causal_attn.h"
#include "../linear.h"

#include <cuda_runtime.h>

CausalAttention::CausalAttention(int n_embd, int n_heads)
    : _n_embd(n_embd), _n_heads(n_heads) {

    _proj_weights.resize(3 * static_cast<size_t>(n_embd) * static_cast<size_t>(n_heads));
    
    // initialize softmax layer
    Softmax softmax;

    // initialize linear layer
    LinearLayer proj_weights_layer(n_embd, 3 * n_embd * n_heads);
    proj_weights_layer.set_weights(
        thrust::raw_pointer_cast(_proj_weights.data()),
        _proj_weights.size()
    );

    // initialize attention layer
    Attention attention(n_embd, n_heads);

    // initialize mask
    _mask.resize(static_cast<size_t>(n_embd) * static_cast<size_t>(n_heads));
}

void CausalAttention::forward(const Tensor& input, Tensor& output) {
    // input shape is [batch_size, seq_len, n_embd]
    // output shape is [batch_size, seq_len, n_embd]

    // project input -> query_key_value using LinearLayer
    Tensor query_key_value;
    proj_weights_layer.forward(input, query_key_value);
    
    // Split into Q, K, V
    std::vector<Tensor> query_key_value_parts = query_key_value.split(3, 2);
    Tensor query = query_key_value_parts[0];
    Tensor key = query_key_value_parts[1];
    Tensor value = query_key_value_parts[2];

    // apply attention to query, key, and value
    // and masking.
    // softmax(query @ key^T) * value / sqrt(d_k)
    // output shape is [batch_size, seq_len, n_embd]
    attention.forward(query, key, value, output);
}

Attention::Attention(int n_embd, int n_heads)
    : _n_embd(n_embd), _n_heads(n_heads) {
        _softmax = Softmax();
}

void Attention::forward(
    const Tensor& query, 
    const Tensor& key, 
    const Tensor& value,
    Tensor& output) {
    // query_key_value shape is [batch_size, seq_len, 3 * n_embd]
    // output shape is [batch_size, seq_len, n_embd]
    
    // create attention scores tensor
    _attention_scores.resize({query.shape(0), query.shape(1), key.shape(1)});

    int block = 256;
    int grid = (_attention_scores.size() + block - 1) / block;
    attn_scores_kernel<<<grid, block>>>(
        query.data_ptr(),
        key.data_ptr(),
        _attention_scores.data_ptr(),
        query.shape(0),
        query.shape(1),
        key.shape(1)
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
    int block = 256;
    int grid = (output.size() + block - 1) / block;

    matmul_kernel<<<grid, block>>>(
        _attention_scores.data_ptr(),
        value.data_ptr(),
        output.data_ptr(),
        output.shape(0),
        output.shape(1),
        output.shape(2),
        value.shape(2)
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
}

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
            res += query[batch_idx * m * n + row * n + i] * key[batch_idx * n * k + i * k + col];
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