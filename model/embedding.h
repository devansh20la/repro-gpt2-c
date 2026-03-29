#pragma once

#include <cstdint>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
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

    // Weight tying helper:
    // Returns a raw device pointer to the embedding table laid out as
    //   float[num_embeddings, out_feature_size] (row-major).
    //
    // This matches LinearLayer's weight layout when:
    //   out_features == num_embeddings  and  in_features == out_feature_size.
    const float* weights_ptr() const {
        return thrust::raw_pointer_cast(_embeddings.data());
    }

    size_t weights_size() const { return _embeddings.size(); }
};
