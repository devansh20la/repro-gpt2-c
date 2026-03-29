#pragma once

#include "../tensor.h"
#include "norm.h"
#include "causal_attn.h"
#include "mlp.h"

class Block {
    private:
        int _in_features;
        int _scaling_factor;
        LayerNorm _ln1;
        CausalMultiHeadedAttention _attn;
        LayerNorm _ln2;
        MLP _mlp;
    public:
        Block(int in_features, int scaling_factor, int n_heads);
        void forward(const Tensor& input, Tensor& output);
        void init_weights(float mean, float std);
};