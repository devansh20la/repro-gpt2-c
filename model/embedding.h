#pragma once

#include <cstdint>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "../tensor.h"

// PyTorch: `nn.Embedding(num_embeddings, embedding_dim)`
// Lookup: for each token id t, output[..., :] = embedding_table[t, :].
class Embedding {
private:
    int _num_embeddings;
    int _out_feature_size;
    thrust::device_vector<float> _embeddings;  // [num_embeddings, out_feature_size] row-major

public:
    Embedding(int num_embeddings, int out_feature_size);

    // input:  TensorBase<uint16_t> [B, N]  (token ids; dtype matches data/preprocess.py)
    // output: float Tensor [B, N, out_feature_size]
    // forward pass the input through the embedding layer
    void forward(const TensorBase<uint16_t>& input, Tensor& output) const;

    // initialize the embeddings of the embedding layer
    void init_embeddings(float mean, float std);

    // Weight tying with `LinearLayer(vocab_size, n_embd)`:
    // same layout as LinearLayer weights when out_features == num_embeddings and in_features == out_feature_size.
    const float* weights_ptr() const {
        return thrust::raw_pointer_cast(_embeddings.data());
    }

    size_t weights_size() const { return _embeddings.size(); }

    // set the weights of the embedding layer
    void set_weights(const float* weights, size_t size);
};
