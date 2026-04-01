#include "../act.h"

#include <cuda_runtime.h>

__global__ void relu_forward_kernel() {
    // Cuda kernel to apply the ReLU activation function.
    // 1. Get the index of the current thread
    // 2. Check if the index is within the range of the output
    // 3. Apply the ReLU activation function
    // 4. Return the result
}

__global__ void gelu_forward_kernel() {
    // Cuda kernel to apply the GELU activation function.
    // 1. Get the index of the current thread
    // 2. Check if the index is within the range of the output
    // 3. Apply the GELU activation function
    // 4. Return the result
}


__global__ void softmax_forward_kernel() {
    // Note that you can split this kernel into multiple separate kernels
    // for more speedup. However, the easiest way to implement is to use single thread to compute full 
    // softmax for the last dimension.


    // Cuda kernel to apply the Softmax activation function.
    // 1. Get the index of the current thread
    // 2. Check if the index is within the range of the output
    // 3. Apply the Softmax activation function
    // 4. Return the result
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
    // Cuda kernel to apply
}
