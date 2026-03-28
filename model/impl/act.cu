#include "../act.h"

#include <cuda_runtime.h>

__global__ void relu_forward_kernel(const float* input, float* output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        const float x = input[idx];
        output[idx] = (x > 0.0f) ? x : 0.0f;
    }
}

// GELU activation using GPT-2's tanh approximation.
//
// The exact GELU is:    GELU(x) = x * Φ(x)    where Φ is the normal CDF.
// Computing the normal CDF on a GPU is expensive (it uses erf()),
// so GPT-2 uses a tanh approximation that's faster and nearly identical:
//
//   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// Breaking it down for one element:
//   1. Compute the cubic term:    inner = x + 0.044715 * x³
//   2. Scale by √(2/π) ≈ 0.7978:  inner = 0.7978 * inner
//   3. Apply tanh:                 t = tanh(inner)     → range (-1, 1)
//   4. Shift to (0, 1):           gate = 0.5 * (1 + t)
//   5. Multiply by input:         output = x * gate
//
// For large positive x: tanh → 1, gate → 1, output ≈ x  (pass-through)
// For large negative x: tanh → -1, gate → 0, output ≈ 0  (killed)
// Around x=0: smooth transition (unlike ReLU's sharp corner)
__global__ void gelu_forward_kernel(const float* input, float* output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        const float x = input[idx];
        // 0.797884... = √(2/π)
        const float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
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

void GELU::forward(const Tensor& input, Tensor& output) {
    output.resize(input.shape());

    const int n = static_cast<int>(input.size());
    const int block = 256;
    const int grid = (n + block - 1) / block;

    gelu_forward_kernel<<<grid, block>>>(
        input.data_ptr(),
        output.data_ptr(),
        n);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("GELU::forward kernel launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("GELU::forward kernel execution failed: ") +
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
