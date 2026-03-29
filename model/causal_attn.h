#pragma once

// Attention needs Softmax, CausalMultiHeadedAttention needs LinearLayer + Attention.
// Both need Tensor.
#include "act.h"
#include "linear.h"
#include "../tensor.h"

// ─── Attention ──────────────────────────────────────────────────────────
//
// Scaled dot-product attention with a causal mask:
//
//   scores = Q @ K^T / sqrt(d_head)     [B, H, T, T]  (per head)
//   scores = causal_mask(scores)         future positions → -inf
//   scores = softmax(scores)            [B, T, T]
//   output  = scores @ V               [B, T, C]
//
// This class must be defined BEFORE CausalMultiHeadedAttention because C++ requires
// the full class definition (not just a forward declaration) when a class
// is used as a data member.

class Attention {
private:
    int _n_embd;
    int _n_heads;
    Softmax _softmax;
    Tensor _attention_scores;

public:
    // head_dim = n_embd / n_heads (per-head feature size d_k = d_v).
    Attention(int head_dim, int n_heads);

    // query, key, value: [batch_size, n_heads, seq_len, head_dim]
    // output:            [batch_size, n_heads, seq_len, head_dim]
    void forward(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        Tensor& output
    );
};

// ─── CausalMultiHeadedAttention ────────────────────────────────────────────────────
//
// Full causal multi-head self-attention (GPT-2 style):
//   1. c_attn: [B, T, n_embd] → [B, T, 3*n_embd]  (Q, K, V concatenated)
//   2. Reshape to [B, T, n_heads, head_dim], transpose to [B, n_heads, T, head_dim]
//   3. Per-head scaled dot-product attention + causal mask
//   4. c_proj: [B, T, n_embd] → [B, T, n_embd]
//
// In GPT-2 terms: c_attn (the combined QKV projection) + attention.

class CausalMultiHeadedAttention {
private:
    int _n_embd;
    int _n_heads;    // number of heads
    int _head_dim;   // n_embd / n_heads

    // c_attn: [B, T, n_embd] → [B, T, 3*n_embd]
    LinearLayer _proj;

    // c_proj: concat heads [B, T, n_embd] → [B, T, n_embd]
    LinearLayer _out_proj;

    // Per-head attention (scores + softmax + @ V)
    Attention _attention;

public:
    CausalMultiHeadedAttention(int n_embd, int n_heads);

    // input:  [batch_size, seq_len, n_embd]
    // output: [batch_size, seq_len, n_embd]
    void forward(const Tensor& input, Tensor& output);

    void init_weights(float mean, float std);
};
