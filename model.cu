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
#include <cstdint>
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

        // Initialize the weights of the model
        init_weights(0.0f, 1.0f);
}

void GPT2::init_weights(float mean, float std) {
    // Although this function is not necessary, since we are not training the model. 
    // it is still a good practice to initialize the weights of the model.
    // for cases when we want to pass a random input to the model.

    _embedding1.init_embeddings(mean, std);
    _embedding2.init_embeddings(mean, std);
    
    for (int i = 0; i < _config.n_layers; i++) {
        _blocks[i].init_weights(mean, std);
    }

    // _linear_out is a linear layer that is used to output the logits.
    // The weights of this layer need to be tied to the weights of the embedding layer.
    // implement the weight tying here.
    _linear_out


    _ln_out.init_weights({_config.n_embd}, mean, std);
}

// tok: [B, T, C], pos: [1, T, C] (same positions for every batch row)
__global__ void add_broadcast_pos_kernel() {
    // Cuda kernel to add the token embedding and position embedding.
    // 1. Get the index of the current thread
    // 2. Check if the index is within the range of the output
    // 3. Add the token embedding and position embedding
    // 4. Return the result
}

void GPT2::forward(const TensorBase<uint16_t>& input, Tensor& output) {
    // Get the input shape i.e [batch_size, sequence_length] / [B, T] (token ids on GPU)
    const int B = input.shape(0);
    const int T = input.shape(1);

    // Get token embedding from the embedding layer

    // Initialize the position embedding tensor

    // Get the position embedding from the embedding layer

    // Add token embedding and position embedding (broadcast pos across batch)
    // using the cuda kernel add_broadcast_pos_kernel

    // checks for errors in the cuda kernel
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

    // Pass through transformer blocks
    for (int i = 0; i < _config.n_layers; i++) {
        _blocks[i].forward(embed_out, embed_out);
    }
    
    // layer norm & linear out layer
    // Note that the linear out layer is the same as the embedding layer.
    // so we need to perform the weight tying for the linear out layer.

}


void GPT2::load_weights(const std::string& path) {
    // Function to load the weights of the model from a file
    // 1. Open the file path as a binary file
        // Look at the code and formate in utils/process_hf_weights.py 
        // to understand the format of the weights file.
    // 2. Read the weights from the file
    // 3. Load the weights into the model
    // 4. Close the file.
}