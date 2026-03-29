#include "model.h"
#include "model/impl/embedding.cu"  
#include "model/impl/linear.cu"
#include "model/impl/norm.cu"
#include "model/impl/act.cu"
#include "model/impl/causal_attn.cu"
#include "model/impl/block.cu"
#include "model/impl/mlp.cu"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

GPT2::GPT2(const GPT2Config& config)
    : _config(config),
      _embedding1(config.vocab_size, config.n_embd),
      _embedding2(config.block_size, config.n_embd),
      _blocks(config.n_layers, Block(config.n_embd, config.scaling_factor, config.n_head)),
      _linear_out(config.n_embd, config.vocab_size), // need to perform weight tying here
      _ln_out({config.n_embd}) {
        init_weights(0.0f, 1.0f);
}

void GPT2::init_weights(float mean, float std) {
    _embedding1.init_embeddings(mean, std);
    _embedding2.init_embeddings(mean, std);
    for (int i = 0; i < _config.n_layers; i++) {
        _blocks[i].init_weights(mean, std);
    }
    _linear_out.init_weights(mean, std);
    _linear_out.set_weights(_embedding1.weights_ptr(), _embedding1.weights_size());
    _ln_out.init_weights({_config.n_embd}, mean, std);
}

// tok: [B, T, C], pos: [1, T, C] (same positions for every batch row)
__global__ void add_broadcast_pos_kernel(
    const float* tok,
    const float* pos,
    float* out,
    int B,
    int T,
    int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * T * C;
    if (idx < total) {
        const int pos_idx = idx % (T * C);
        out[idx] = tok[idx] + pos[pos_idx];
    }
}

void GPT2::forward(const TensorBase<uint16_t>& input, Tensor& output) {
    // input shape is [batch_size, sequence_length] / [B, T] (token ids on GPU)

    // Get token embedding
    Tensor tok_embed({input.shape(0), input.shape(1), _config.n_embd});
    _embedding1.forward(input, tok_embed);

    // Position ids [1, T] — one sequence 0..T-1; same for every batch row
    const int T = input.shape(1);
    TensorBase<uint16_t> pos_ids({1, T});
    {
        thrust::host_vector<uint16_t> h(static_cast<size_t>(T));
        for (int t = 0; t < T; ++t) {
            h[static_cast<size_t>(t)] = static_cast<uint16_t>(t);
        }
        thrust::copy(
            h.begin(),
            h.end(),
            thrust::device_ptr<uint16_t>(pos_ids.data_ptr()));
    }

    Tensor pos_embed_out({1, T, _config.n_embd});
    _embedding2.forward(pos_ids, pos_embed_out);

    // Add token embedding and position embedding (broadcast pos across batch)
    Tensor embed_out({input.shape(0), T, _config.n_embd});
    const int B = input.shape(0);
    const int C = _config.n_embd;
    int block = 256;
    int grid = (static_cast<int>(embed_out.size()) + block - 1) / block;
    add_broadcast_pos_kernel<<<grid, block>>>(
        tok_embed.data_ptr(),
        pos_embed_out.data_ptr(),
        embed_out.data_ptr(),
        B,
        T,
        C);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("add_broadcast_pos_kernel launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("add_broadcast_pos_kernel execution failed: ") +
            cudaGetErrorString(sync_err));
    }

    // pass through transformer blocks
    for (int i = 0; i < _config.n_layers; i++) {
        _blocks[i].forward(embed_out, embed_out);
    }

    // layer norm & linear
    Tensor ln_out_output({embed_out.shape(0), embed_out.shape(1), _config.n_embd});
    _ln_out.forward(embed_out, ln_out_output);

    // output shape is [B, T, vocab_size]
    // need to perform weight tying for this linear layer
    output.resize({input.shape(0), input.shape(1), _config.vocab_size});
    _linear_out.forward(ln_out_output, output);
}