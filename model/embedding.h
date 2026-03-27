#pragma once

#include <cstdint>
#include "../tensor.h"

class Embedding {
private:
    int _num_embeddings;
    int _out_feature_size;
    thrust::device_vector<float> _embeddings;  // [_num_embeddings, _out_feature_size]

public:
    Embedding(int num_embeddings, int out_feature_size);

    // input:  TensorBase<uint16_t> shaped [B, N]  (batch of token id sequences)
    // output: Tensor shaped [B, N, out_feature_size]
    void forward(const TensorBase<uint16_t>& input, Tensor& output) const;

    void init_embeddings(float mean, float std);
};
