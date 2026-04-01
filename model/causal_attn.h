#pragma once

#include "act.h"
#include "linear.h"
#include "../tensor.h"
#include <string>
#include <unordered_map>
#include <vector>


class Attention {
    // Layer to implement the attention mechanism.
    // Before implementing this, you should implement the softmax layer.
    // Note that the attention mechanism is a key component of the transformer architecture.
    // This function only implements the attention mechanism, not the causal mask.
    // This is also single headed attention, not multi-headed attention.
    // Multi-headed attention is implemented in the CausalMultiHeadedAttention layer.
    private:
        int _n_embd;
        int _n_heads;
        Softmax _softmax;
        Tensor _attention_scores;

    public:
        Attention(int head_dim, int n_heads);

        void forward(
            const Tensor& query,
            const Tensor& key,
            const Tensor& value,
            Tensor& output);
};

// Causal multi-head self-attention (GPT-2):
//   1. c_attn: [B, T, n_embd] -> [B, T, 3*n_embd]  (Q, K, V stacked)
//   2. Split + reshape + transpose to [B, H, T, d]
//   3. Attention (masked) -> [B, H, T, d]
//   4. Merge heads -> [B, T, n_embd]; c_proj -> [B, T, n_embd]
class CausalMultiHeadedAttention {
    private:
        int _n_embd;
        int _n_heads;
        int _head_dim;

        LinearLayer _proj;     // c_attn
        LinearLayer _out_proj;   // c_proj
        Attention _attention;

    public:
        CausalMultiHeadedAttention(int n_embd, int n_heads);

        // input / output: [B, T, n_embd]
        void forward(const Tensor& input, Tensor& output);

        // initialize the weights of the causal multi-head attention layer.
        void init_weights(float mean, float std);

        // load the weights of the causal multi-head attention layer from a map.
        void load_weights(
            const std::unordered_map<std::string, std::vector<float>>& tensors,
            const std::string& prefix);
};
