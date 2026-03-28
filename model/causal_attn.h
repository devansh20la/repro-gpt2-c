#pragma once

// Attention needs Softmax, CausalAttention needs LinearLayer + Attention.
// Both need Tensor.
#include "act.h"
#include "linear.h"
#include "../tensor.h"

// ─── Attention ──────────────────────────────────────────────────────────
//
// Scaled dot-product attention with a causal mask:
//
//   scores = Q @ K^T / sqrt(C)          [B, T, T]
//   scores = causal_mask(scores)         future positions → -inf
//   scores = softmax(scores)            [B, T, T]
//   output  = scores @ V               [B, T, C]
//
// This class must be defined BEFORE CausalAttention because C++ requires
// the full class definition (not just a forward declaration) when a class
// is used as a data member.

class Attention {
private:
    int _n_embd;
    Softmax _softmax;
    Tensor _attention_scores;

public:
    Attention(int n_embd);

    // query, key, value: [batch_size, seq_len, n_embd]
    // output:            [batch_size, seq_len, n_embd]
    void forward(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        Tensor& output
    );
};

// ─── CausalAttention ────────────────────────────────────────────────────
//
// Full causal self-attention block:
//   1. Linear projection: [B, T, C] → [B, T, 3C]   (produces Q, K, V)
//   2. Split into Q, K, V: each [B, T, C]
//   3. Scaled dot-product attention with causal mask
//
// In GPT-2 terms: c_attn (the combined QKV projection) + attention.

class CausalAttention {
private:
    int _n_embd;

    // The QKV projection:  [B, T, n_embd] → [B, T, 3 * n_embd]
    LinearLayer _proj;

    // The attention computation (scores + softmax + value multiplication)
    Attention _attention;

public:
    CausalAttention(int n_embd);

    // input:  [batch_size, seq_len, n_embd]
    // output: [batch_size, seq_len, n_embd]
    void forward(const Tensor& input, Tensor& output);

    void init_weights(float mean, float std);
};
