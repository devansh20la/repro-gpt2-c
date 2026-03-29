#include "../causal_attn.h"

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

// Q,K,V after split are [B, T, H, d]; attention uses [B, H, T, d].
__global__ void transpose_bthd_to_bhtd_kernel(
    const float* src,
    float* dst,
    int B,
    int T,
    int H,
    int d) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * T * H * d;
    if (idx < total) {
        const int i = idx % d;
        int x = idx / d;
        const int h = x % H;
        x /= H;
        const int t = x % T;
        const int b = x / T;
        const int dst_idx = b * H * T * d + h * T * d + t * d + i;
        dst[dst_idx] = src[idx];
    }
}

// Merge heads: [B, H, T, d] -> [B, T, H, d] then caller reshapes to [B, T, H*d].
__global__ void transpose_bhtd_to_bthd_kernel(
    const float* src,
    float* dst,
    int B,
    int H,
    int T,
    int d) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * H * T * d;
    if (idx < total) {
        const int i = idx % d;
        int x = idx / d;
        const int t = x % T;
        x /= T;
        const int h = x % H;
        const int b = x / H;
        const int dst_idx = b * T * H * d + t * H * d + h * d + i;
        dst[dst_idx] = src[idx];
    }
}

__global__ void attn_scores_kernel(
    const float* query,
    const float* key,
    float* output,
    int batch_size,
    int n_heads,
    int seq_len,
    int n_embd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * n_heads * seq_len * seq_len) {
        const int batch_idx = idx / (n_heads * seq_len * seq_len);
        const int local_idx = idx % (n_heads * seq_len * seq_len);
        const int head = local_idx / (seq_len * seq_len);
        const int rem = local_idx % (seq_len * seq_len);
        const int row = rem / seq_len;
        const int col = rem % seq_len;
        
        float res = 0.0f;
        float query_val = 0.0f;
        float key_val = 0.0f;
        for (int i = 0; i < n_embd; i++) {
            query_val = query[batch_idx * n_heads * seq_len * n_embd + head * seq_len * n_embd + row * n_embd + i];
            key_val = key[batch_idx * n_heads * seq_len * n_embd + head * seq_len * n_embd + col * n_embd + i];
            res += query_val * key_val;
        }

        if (row < col) {
            // Use a large negative sentinel for masking (portable across toolchains).
            // Softmax(exp(x)) will treat this as ~0 probability.
            output[idx] = -INFINITY;
        } else {
            output[idx] = res / sqrtf(static_cast<float>(n_embd));
        }
    }
}

__global__ void matmul_kernel(
    const float* matA,
    const float* matB,
    float* matC,
    int batch_size, 
    int n_heads, 
    int seq_len, 
    int n_embd) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * n_heads * seq_len * n_embd) {
        const int batch_idx = idx / (n_heads * seq_len * n_embd);
        const int local_idx = idx % (n_heads * seq_len * n_embd);
        const int head = local_idx / (seq_len * n_embd);
        const int rem = local_idx % (seq_len * n_embd);
        const int row = rem / n_embd;
        const int col = rem % n_embd;

        float res = 0.0f;
        float attention_score = 0.0f;
        float value_val = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            attention_score = matA[batch_idx * n_heads * seq_len * seq_len + head * seq_len * seq_len + row * seq_len + i];
            value_val = matB[batch_idx * n_heads * seq_len * n_embd + head * seq_len * n_embd + i * n_embd + col];

            res += attention_score * value_val;
        }
        matC[idx] = res;
    }
}

static void launch_transpose_bthd_to_bhtd(
    const float* src,
    float* dst,
    int B,
    int T,
    int H,
    int d) {
    const int total = B * T * H * d;

    int block = 256;
    int grid = (total + block - 1) / block;
    transpose_bthd_to_bhtd_kernel<<<grid, block>>>(src, dst, B, T, H, d);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("transpose_bthd_to_bhtd failed: ") +
            cudaGetErrorString(err));
    }
}

static void launch_transpose_bhtd_to_bthd(
    const float* src,
    float* dst,
    int B,
    int H,
    int T,
    int d) {
    const int total = B * H * T * d;
    int block = 256;
    int grid = (total + block - 1) / block;
    transpose_bhtd_to_bthd_kernel<<<grid, block>>>(src, dst, B, H, T, d);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("transpose_bhtd_to_bthd failed: ") +
            cudaGetErrorString(err));
    }
}

CausalMultiHeadedAttention::CausalMultiHeadedAttention(int n_embd, int n_heads)
    : _n_embd(n_embd),
      _n_heads(n_heads),
      _head_dim(n_embd / n_heads),
      _proj(n_embd, 3 * n_embd),
      _out_proj(n_embd, n_embd),
      _attention(n_embd / n_heads, n_heads) {
    if (n_heads <= 0) {
        throw std::invalid_argument(
            "CausalMultiHeadedAttention: need n_heads > 0");
    }
    if (n_embd % n_heads != 0) {
        throw std::invalid_argument(
            "CausalMultiHeadedAttention: n_embd must be divisible by n_heads");
    }
    init_weights(0.0f, 1.0f);
}

void CausalMultiHeadedAttention::forward(const Tensor& input, Tensor& output) {
    const int B = input.shape(0);
    const int T = input.shape(1);
    const int C = input.shape(2);
    if (C != _n_embd) {
        throw std::invalid_argument(
            "CausalMultiHeadedAttention::forward: input last dim must equal n_embd");
    }
    // QKV from c_attn each [B, T, n_embd] → view as [B, T, H, head_dim].

    Tensor query_key_value;
    query_key_value.resize({B, T, 3 * _n_embd});
    _proj.forward(input, query_key_value);

    std::vector<Tensor> parts = query_key_value.split(3, -1);
    Tensor q_flat = parts[0];
    Tensor k_flat = parts[1];
    Tensor v_flat = parts[2];

    std::vector<int> bthd = {B, T, _n_heads, _head_dim};
    q_flat.reshape(bthd);
    k_flat.reshape(bthd);
    v_flat.reshape(bthd);

    Tensor q_bhtd({B, _n_heads, T, _head_dim});
    Tensor k_bhtd({B, _n_heads, T, _head_dim});
    Tensor v_bhtd({B, _n_heads, T, _head_dim});


    launch_transpose_bthd_to_bhtd(
        q_flat.data_ptr(), q_bhtd.data_ptr(), B, T, _n_heads, _head_dim);
    launch_transpose_bthd_to_bhtd(
        k_flat.data_ptr(), k_bhtd.data_ptr(), B, T, _n_heads, _head_dim);
    launch_transpose_bthd_to_bhtd(
        v_flat.data_ptr(), v_bhtd.data_ptr(), B, T, _n_heads, _head_dim);

    Tensor attn_out_bhtd;
    _attention.forward(q_bhtd, k_bhtd, v_bhtd, attn_out_bhtd);

    Tensor attn_bthd({B, T, _n_heads, _head_dim});
    launch_transpose_bhtd_to_bthd(
        attn_out_bhtd.data_ptr(),
        attn_bthd.data_ptr(),
        B,
        _n_heads,
        T,
        _head_dim);

    // Concatenate heads: [B,T,H,head_dim] → [B,T,n_embd]
    std::vector<int> merged = {B, T, _n_embd};
    attn_bthd.reshape(merged);

    _out_proj.forward(attn_bthd, output);
}

Attention::Attention(int head_dim, int n_heads)
    : _n_embd(head_dim),
      _n_heads(n_heads),
      _softmax() {
    if (n_heads <= 0) {
        throw std::invalid_argument("Attention: need n_heads > 0");
    }
    if (head_dim <= 0) {
        throw std::invalid_argument("Attention: need head_dim > 0");
    }
}

void Attention::forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    Tensor& output) {
    // query, key, value: [batch_size, n_heads, seq_len, head_dim]
    // output:            same
    const int batch_size = query.shape(0);
    const int n_heads = query.shape(1);
    const int seq_len = query.shape(2);
    const int n_embd = query.shape(3);

    if (n_heads != _n_heads || n_embd != _n_embd) {
        throw std::invalid_argument(
            "Attention::forward: tensor n_heads / last dim do not match constructor");
    }

    _attention_scores.resize({batch_size, n_heads, seq_len, seq_len});
    output.resize({batch_size, n_heads, seq_len, n_embd});

    const int block_attn = 1024;
    int grid = (static_cast<int>(_attention_scores.size()) + block_attn - 1) / block_attn;
    attn_scores_kernel<<<grid, block_attn>>>(
        query.data_ptr(),
        key.data_ptr(),
        _attention_scores.data_ptr(),
        batch_size,
        n_heads,
        seq_len,
        n_embd
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

    Tensor attention_scores_softmax;
    attention_scores_softmax.resize(_attention_scores.shape());
    _softmax.forward(_attention_scores, attention_scores_softmax);

    // attention_scores_softmax shape is [B, n_heads, seq_len, seq_len]
    // value shape is [B, n_heads, seq_len, n_embd]
    // apply attention scores to value
    const int block_mm = 256;
    grid = (static_cast<int>(output.size()) + block_mm - 1) / block_mm;
    matmul_kernel<<<grid, block_mm>>>(
        attention_scores_softmax.data_ptr(),
        value.data_ptr(),
        output.data_ptr(),
        batch_size,
        n_heads,
        seq_len,
        n_embd);

    launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Attention::forward matmul launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Attention::forward matmul execution failed: ") +
            cudaGetErrorString(sync_err));
    }
}

void CausalMultiHeadedAttention::init_weights(float mean, float std) {
    _proj.init_weights(mean, std);
    _out_proj.init_weights(mean, std);
}
