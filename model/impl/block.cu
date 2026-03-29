#include "../block.h"

#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int B, int T, int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * C) {
        c[idx] = a[idx] + b[idx];
    }
}

Block::Block(int in_features, int scaling_factor, int n_heads)
    : _in_features(in_features),
      _scaling_factor(scaling_factor),
      _ln1({in_features}),
      _attn(in_features, n_heads),
      _ln2({in_features}),
      _mlp(in_features, scaling_factor) {}

void Block::forward(const Tensor& input, Tensor& output) {
    // Forward preserves shape: [B,T,C] -> [B,T,C]
    output.resize(input.shape());

    Tensor _ln1_output({input.shape(0), input.shape(1), _in_features});
    _ln1.forward(input, _ln1_output);

    Tensor _attn_output({input.shape(0), input.shape(1), _in_features});
    _attn.forward(_ln1_output, _attn_output);

    // residual connection
    int block = 256;
    int grid = (input.shape(0) * input.shape(1) * input.shape(2) + block - 1) / block;
    add_kernel<<<grid, block>>>(
        _attn_output.data_ptr(),
        input.data_ptr(),
        _attn_output.data_ptr(),
        input.shape(0),
        input.shape(1),
        input.shape(2)
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("add_kernel launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("add_kernel execution failed: ") +
            cudaGetErrorString(sync_err));
    }

    // Save the post-attention residual stream (x + attn(ln1(x))).
    // We need this for the second residual add after the MLP.
    Tensor attn_resid = _attn_output;

    // layer norm2 & mlp
    Tensor _ln2_output({input.shape(0), input.shape(1), _in_features});
    _ln2.forward(_attn_output, _ln2_output);
    Tensor mlp_out;
    _mlp.forward(_ln2_output, mlp_out);

    // residual connection
    grid = (input.shape(0) * input.shape(1) * input.shape(2) + block - 1) / block;
    add_kernel<<<grid, block>>>(
        mlp_out.data_ptr(),
        attn_resid.data_ptr(),
        output.data_ptr(),
        input.shape(0),
        input.shape(1),
        input.shape(2)
    );

    launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("add_kernel launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("add_kernel execution failed: ") +
            cudaGetErrorString(sync_err));
    }
}

void Block::init_weights(float mean, float std) {
    _ln1.init_weights({_in_features}, mean, std);
    _attn.init_weights(mean, std);
    _ln2.init_weights({_in_features}, mean, std);
    _mlp.init_weights(mean, std);
}