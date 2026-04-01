#include "../causal_attn.h"

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

// ─── Layout transposes ───────────────────────────────────────────────────
// [B, T, H, d] <-> [B, H, T, d] for attention.

__global__ void transpose_bthd_to_bhtd_kernel() {
    // You can work with an intelligent cuda kernel and tiliing strategy without transposing, 
    // However, I highly recommend implementing it for ease of understanding and debugging.
    // 1. Linear index idx over B*T*H*d; map idx -> (b, t, h, i) with i in [0,d).
    // 2. Read src at [b,t,h,i] layout (contiguous row-major for [B,T,H,d]).
    // 3. Write dst at [b,h,t,i] layout ([B,H,T,d]).
    // 4. Use the inverse mapping of your chosen flattening order consistently.
}

__global__ void transpose_bhtd_to_bthd_kernel() {
    // Inverse of transpose_bthd_to_bhtd: [B,H,T,d] -> [B,T,H,d].
}

// ─── Attention scores QK^T / sqrt(d) + causal mask ───────────────────────

__global__ void attn_scores_kernel() {
    // Layout: Q, K are [B, H, T, d]. One thread per (b, h, row, col) in scores [B, H, T, T].
    // 1. Decode idx -> (b, h, row, col).
    // 2. Dot product: sum_i Q[b,h,row,i] * K[b,h,col,i].
    // 3. Scale by 1/sqrt(head_dim).
    // 4. Causal: if col > row, set score to -INFINITY (or large negative) before softmax.
}

// ─── attn @ V ────────────────────────────────────────────────────────────
__global__ void matmul_kernel() {
    // matA: softmax scores [B, H, T, T]; matB: V [B, H, T, d]; matC: [B, H, T, d].
    // For each (b, h, row, col): C[b,h,row,col] = sum_t A[b,h,row,t] * B[b,h,t,col].
}

CausalMultiHeadedAttention::CausalMultiHeadedAttention(int n_embd, int n_heads)
    : _n_embd(n_embd),
      _n_heads(n_heads),
      _head_dim(n_embd / n_heads),
      _proj(n_embd, 3 * n_embd),
      _out_proj(n_embd, n_embd),
      _attention(n_embd / n_heads, n_heads) {
    if (n_heads <= 0) {
        throw std::invalid_argument("CausalMultiHeadedAttention: need n_heads > 0");
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
    // 1. qkv = c_attn(x) -> [B, T, 3*n_embd]

    // 2. Split into Q, K, V along last dim (requires Tensor::split / views).

    // 3. Transpose Q, K, V to [B, H, T, d]

    // 4. Pass Q, K, V to the attention layer

    // 5. Transpose the output of the attention layer to [B, T, n_embd]

    // 6. Pass the output of the attention layer to the output projection layer
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
    const int batch_size = query.shape(0);
    const int n_heads = query.shape(1);
    const int seq_len = query.shape(2);
    const int n_embd = query.shape(3);
    // 1. Perform checks on the query, key, and value shapes
    if (n_heads != _n_heads || n_embd != _n_embd) {
        throw std::invalid_argument(
            "Attention::forward: tensor n_heads / last dim do not match constructor");
    }

    // 2. Resize the output tensor
    // 3. Launch the attention scores kernel
    // 4. Perform checks for errors in the kernel launch and execution
    // 5. Synchronize the device to ensure the kernel has completed.

    
    // 6. Pass the output of the attention scores kernel to the softmax layer
    // 7. Pass the output of the softmax layer to the matrix multiplication kernel
    // 8. Perform checks for errors in the kernel launch and execution
    // 9. Synchronize the device to ensure the kernel has completed
}

void CausalMultiHeadedAttention::init_weights(float mean, float std) {
    _proj.init_weights(mean, std);
    _out_proj.init_weights(mean, std);
}

void CausalMultiHeadedAttention::load_weights(
    const std::unordered_map<std::string, std::vector<float>>& tensors,
    const std::string& prefix) {
    // 1. Load the weights of the projection layer
    // 2. Load the weights of the output projection layer
}
