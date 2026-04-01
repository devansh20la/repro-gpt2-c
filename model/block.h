#pragma once

#include "../tensor.h"
#include "norm.h"
#include "causal_attn.h"
#include "mlp.h"
#include <string>
#include <unordered_map>
#include <vector>

class Block {
    // Layer to implment a GPT2 transformer block.
    // Before implementing this, you should implement the LayerNorm, CausalMultiHeadedAttention, and MLP layers.
    private:
        int _in_features;
        int _scaling_factor;
        LayerNorm _ln1;
        CausalMultiHeadedAttention _attn;
        LayerNorm _ln2;
        MLP _mlp;
    public:
        // Constructor to initialize the block layer
        Block(int in_features, int scaling_factor, int n_heads);

        // forward pass the input through the block layer
        void forward(const Tensor& input, Tensor& output);

        // initialize the weights of the block layer
        void init_weights(float mean, float std);

        // load the weights of the block layer from a file
        // Note that the weights of the block layer are stored in a file in the same format as the weights of the layer norm and causal multi-head attention and MLP layers.
        // You should implement the load_weights function in the LayerNorm, CausalMultiHeadedAttention, and MLP layers before implementing this.
        void load_weights(
            const std::unordered_map<std::string, std::vector<float>>& tensors,
            const std::string& prefix);
};