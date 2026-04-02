#include "model.h"
#include "model/impl/embedding.cu"  
#include "model/impl/linear.cu"
#include "model/impl/norm.cu"
#include "model/impl/act.cu"
#include "model/impl/causal_attn.cu"
#include "model/impl/block.cu"
#include "model/impl/mlp.cu"

#include <nvtx3/nvtx3.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
    nvtx3::scoped_range const gpt2_forward{"GPT2/forward"};

    // input shape is [batch_size, sequence_length] / [B, T] (token ids on GPU)
    const int B = input.shape(0);
    const int T = input.shape(1);

    Tensor tok_embed({B, T, _config.n_embd});
    {
        nvtx3::scoped_range const r{"GPT2/embedding_token"};
        _embedding1.forward(input, tok_embed);
    }

    // Position ids [1, T] — one sequence 0..T-1; same for every batch row
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
    {
        nvtx3::scoped_range const r{"GPT2/embedding_position"};
        _embedding2.forward(pos_ids, pos_embed_out);
    }

    // Add token embedding and position embedding (broadcast pos across batch)
    Tensor embed_out({B, T, _config.n_embd});
    {
        nvtx3::scoped_range const r{"GPT2/add_token_and_position"};
        int block = 256;
        int grid = (static_cast<int>(embed_out.size()) + block - 1) / block;
        add_broadcast_pos_kernel<<<grid, block>>>(
            tok_embed.data_ptr(),
            pos_embed_out.data_ptr(),
            embed_out.data_ptr(),
            B,
            T,
            _config.n_embd);

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
    }

    for (int i = 0; i < _config.n_layers; i++) {
        char block_label[48];
        std::snprintf(block_label, sizeof(block_label), "GPT2/transformer_block/%d", i);
        nvtx3::scoped_range const block_range{block_label};
        _blocks[i].forward(embed_out, embed_out);
    }

    Tensor ln_out_output({embed_out.shape(0), embed_out.shape(1), _config.n_embd});
    {
        nvtx3::scoped_range const r{"GPT2/layer_norm_final"};
        _ln_out.forward(embed_out, ln_out_output);
    }

    output.resize({input.shape(0), input.shape(1), _config.vocab_size});
    {
        nvtx3::scoped_range const r{"GPT2/lm_head_linear"};
        _linear_out.forward(ln_out_output, output);
    }
}


void GPT2::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(
            std::string("Failed to open weights file: ") + path);
    }

    auto read_exact = [&](void* dst, size_t n) {
        // read n bytes from the file into dst
        file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
        if (file.gcount() != static_cast<std::streamsize>(n)) {
            throw std::runtime_error("Unexpected EOF while reading weights");
        }
    };

    // get total number of tensor weights, this should same as
    // the number of parameters in the model
    uint32_t count = 0;
    read_exact(&count, sizeof(count));

    // Store all tensors as host float vectors keyed by name.
    std::unordered_map<std::string, std::vector<float>> tensors;
    tensors.reserve(static_cast<size_t>(count));

    for (uint32_t i = 0; i < count; i++) {
        uint32_t key_len = 0;
        read_exact(&key_len, sizeof(key_len));
        std::string key;
        key.resize(key_len);
        if (key_len) {
            read_exact(key.data(), key_len);
        }

        uint32_t dtype = 0;
        read_exact(&dtype, sizeof(dtype));
        if (dtype != 1) {
            throw std::runtime_error("Unsupported dtype code in weights (expected 1=float32)");
        }

        uint32_t ndim = 0;
        read_exact(&ndim, sizeof(ndim));
        std::vector<int32_t> shape;
        shape.resize(ndim);
        if (ndim) {
            read_exact(shape.data(), ndim * sizeof(int32_t));
        }

        uint64_t data_nbytes = 0;
        read_exact(&data_nbytes, sizeof(data_nbytes));

        const size_t n_floats = static_cast<size_t>(data_nbytes / sizeof(float));
        std::vector<float> data;
        data.resize(n_floats);
        if (n_floats) {
            read_exact(data.data(), static_cast<size_t>(data_nbytes));
        }

        tensors.emplace(std::move(key), std::move(data));
    }

    // Top-level tensors
    {
        const auto& wte = tensors.at("embedding1.weights");
        const auto& wpe = tensors.at("embedding2.weights");
        _embedding1.set_weights(wte.data(), wte.size());
        _embedding2.set_weights(wpe.data(), wpe.size());
    }

    {
        for (int bi = 0; bi < _config.n_layers; bi++) {
            _blocks[bi].load_weights(tensors, "blocks." + std::to_string(bi) + ".");
        }
    }
    
    {
        const auto& lnw = tensors.at("ln_out.weight");
        const auto& lnb = tensors.at("ln_out.bias");
        _ln_out.set_params(lnw.data(), lnw.size(), lnb.data(), lnb.size());
    }

    {
        const auto& lmh = tensors.at("linear_out.weights");
        _linear_out.set_weights(lmh.data(), lmh.size());
    }

    // linear_out.weights is tied to embedding1 in init_weights().
    // If you want to verify, you can compare tensors["linear_out.weights"] vs embedding1.weights.
}