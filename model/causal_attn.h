#pragma once

#include "../tensor.h"
#include <thrust/device_vector.h>


class CausalAttention {
    private:
        int _n_embd;
        int _n_heads;

        thrust::device_vector<float> _proj_weights;
        thrust::device_vector<float> _mask;
    public:
        CausalAttention(int n_embd, int n_heads);

        // input:  [batch_size, seq_len, n_embd]
        // output: [batch_size, seq_len, n_embd]
        void forward(const Tensor& input, Tensor& output);

        void init_weights(float mean, float std);
};

class Attention {
    private:
        int _n_embd;
        int _n_heads;
        Softmax _softmax;
        Tensor _attention_scores;

    public:
        Attention(int n_embd, int n_heads);

        // query_key_value: [batch_size, seq_len, 3 * n_embd]
        // output:          [batch_size, seq_len, n_embd]
        void forward(
            const Tensor& query, 
            const Tensor& key, 
            const Tensor& value, 
            Tensor& output
        );
};
