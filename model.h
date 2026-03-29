#pragma once

#include <cstdint>

#include "tensor.h"

#include "model/embedding.h"
#include "model/linear.h"
#include "model/norm.h"
#include "model/causal_attn.h"
#include "model/block.h"
#include "model/mlp.h"
#include "gpt2_config.h"

class GPT2 {
    private:
        GPT2Config _config;
        Embedding _embedding1;
        Embedding _embedding2;
        std::vector<Block> _blocks;
        LinearLayer _linear_out;
        LayerNorm _ln_out;
    
    public:
        GPT2(const GPT2Config& config);
        void forward(const TensorBase<uint16_t>& input, Tensor& output);
        void init_weights(float mean, float std);
};