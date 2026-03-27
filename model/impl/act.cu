#include "../act.h"

#include <cuda_runtime.h>

__global__ void relu_forward_kernel(const float* input, float* output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        const float x = input[idx];
        output[idx] = (x > 0.0f) ? x : 0.0f;
    }
}

// Each thread handles one row of length `n`.
// row_idx = which row this thread is responsible for.
// offset  = row_idx * n = where this row starts in the flat buffer.
//
// Step 1: find max for numerical stability (prevents exp overflow)
// Step 2: exp(x_i - max) for each element
// Step 3: divide by sum
__global__ void softmax_forward_kernel(const float* input, float* output,
                                       int n, int num_rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        const int offset = row * n;

        // Step 1: find max value in this row
        float max_val = input[offset];
        for (int i = 1; i < n; i++) {
            max_val = fmaxf(max_val, input[offset + i]);
        }

        // Step 2: exp(x_i - max) and accumulate sum
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            output[offset + i] = expf(input[offset + i] - max_val);
            sum += output[offset + i];
        }

        // Step 3: normalize
        for (int i = 0; i < n; i++) {
            output[offset + i] /= sum;
        }
    }
}

void ReLU::forward(const Tensor& input, Tensor& output) {
    output.resize(input.shape());

    const int n = static_cast<int>(input.size());
    const int block = 256;
    const int grid = (n + block - 1) / block;

    relu_forward_kernel<<<grid, block>>>(
        input.data_ptr(),
        output.data_ptr(),
        n);
    
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("ReLU::forward kernel launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("ReLU::forward kernel execution failed: ") +
            cudaGetErrorString(sync_err));
    }  
}

void Softmax::forward(const Tensor& input, Tensor& output) {
    output.resize(input.shape());

    // n = size of the last dimension (the axis we softmax over)
    // num_rows = everything else collapsed (product of all other dims)
    // e.g. input [B, T, vocab_size] -> n=vocab_size, num_rows=B*T
    const int n = input.shape(-1);
    const int num_rows = static_cast<int>(input.size()) / n;

    const int block = 256;
    const int grid = (num_rows + block - 1) / block;

    softmax_forward_kernel<<<grid, block>>>(
        input.data_ptr(),
        output.data_ptr(),
        n,
        num_rows);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Softmax::forward kernel launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Softmax::forward kernel execution failed: ") +
            cudaGetErrorString(sync_err));
    }
}
