#include "../act.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

__global__ void relu_forward_kernel(const float* input, float* output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float x = input[idx];
        output[idx] = (x > 0.0f) ? x : 0.0f;
    }
}

void ReLU::forward(const thrust::device_vector<float>& input,
                   thrust::device_vector<float>& output) {
    output.resize(input.size());

    const int n = static_cast<int>(input.size());
    const int block = 256;
    const int grid = (n + block - 1) / block;

    relu_forward_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(input.data()),
        thrust::raw_pointer_cast(output.data()),
        n);
    
    cudaDeviceSynchronize();
}
