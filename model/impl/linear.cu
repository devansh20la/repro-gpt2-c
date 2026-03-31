#include "../linear.h"

#include <random>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

// The kernel is unchanged — it only deals with raw float pointers,
// so it doesn't know or care about Tensor vs device_vector.
__global__ void linear_forward_kernel(const float* input,
                                      const float* weights,
                                      const float* biases,
                                      float* output,
                                      int batch_size,
                                      int in_feature_size,
                                      int out_feature_size,
                                      bool bias) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * out_feature_size) {
        const int batch_idx = idx / out_feature_size;
        const int column_idx = idx % out_feature_size;

        float acc = bias ? biases[column_idx] : 0.0f;

        for (int i = 0; i < in_feature_size; ++i) {
            // weights is laid out as [out_feature_size, in_feature_size] (row-major):
            // weight[out_j, in_i] -> weights[out_j * in_feature_size + in_i]
            acc += input[batch_idx * in_feature_size + i] *
                   weights[column_idx * in_feature_size + i];
        }
        output[batch_idx * out_feature_size + column_idx] = acc;
    }
}

LinearLayer::LinearLayer(int in_features, int out_features, bool bias)
    : _in_features(in_features), _out_features(out_features), _bias(bias) {
    _weights.resize(static_cast<size_t>(_in_features) * static_cast<size_t>(_out_features));
    if (bias) {
        _biases.resize(static_cast<size_t>(_out_features));
    }
    init_weights(0.0f, 1.0f);
}

void LinearLayer::forward(const Tensor& input, Tensor& output) {
    if (input.ndim() < 1) {
        throw std::invalid_argument(
            "LinearLayer::forward: input must have at least 1 dimension");
    }

    const int in_feat = input.shape(-1);

    if (in_feat != _in_features) {
        throw std::invalid_argument(
            "LinearLayer::forward: input's last dimension (" +
            std::to_string(in_feat) + ") must equal in_features (" +
            std::to_string(_in_features) + ")");
    }

    // batch_size = product of all dims except the last
    int batch_size = 1;
    for (int i = 0; i < input.ndim() - 1; i++) {
        batch_size *= input.shape(i);
    }

    // Output shape keeps the leading dims, replaces the last with out_features.
    //   [B, T, C_in] → [B, T, C_out]
    std::vector<int> out_shape;
    for (int i = 0; i < input.ndim() - 1; i++) {
        out_shape.push_back(input.shape(i));
    }
    out_shape.push_back(_out_features);
    output.resize(out_shape);

    const int block = 256;
    const int grid = (_out_features * batch_size + block - 1) / block;

    // Before: thrust::raw_pointer_cast(input.data())  — verbose
    // Now:    input.data_ptr()                         — same raw float*, cleaner

    linear_forward_kernel<<<grid, block>>>(
        input.data_ptr(),
        thrust::raw_pointer_cast(_weights.data()),
        _bias ? thrust::raw_pointer_cast(_biases.data()) : nullptr,
        output.data_ptr(),
        batch_size,
        _in_features,
        _out_features,
        _bias
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("LinearLayer::forward kernel launch failed: ") +
            cudaGetErrorString(launch_err));
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("LinearLayer::forward kernel execution failed: ") +
            cudaGetErrorString(sync_err));
    }
}

void LinearLayer::init_weights(float mean, float std) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(mean, std);

    thrust::host_vector<float> h_w(_weights.size());
    for (size_t i = 0; i < h_w.size(); ++i) h_w[i] = dist(rng);
    thrust::copy(h_w.begin(), h_w.end(), _weights.begin());
    if (_bias) {
        thrust::host_vector<float> h_b(_biases.size());
        for (size_t i = 0; i < h_b.size(); ++i) h_b[i] = 0.0f;
        thrust::copy(h_b.begin(), h_b.end(), _biases.begin());
    }
}

void LinearLayer::set_weights(const float* weights, size_t size) {
    if (size != _weights.size()) {
        throw std::invalid_argument("LinearLayer::set_weights: size must equal weights.size()");
    }
    thrust::copy(weights, weights + size, _weights.begin());
}

void LinearLayer::set_biases(const float* biases, size_t size) {
    if (!_bias) {
        throw std::invalid_argument("LinearLayer::set_biases: layer was constructed with bias=false");
    }
    if (size != _biases.size()) {
        throw std::invalid_argument("LinearLayer::set_biases: size must equal biases.size()");
    }
    thrust::copy(biases, biases + size, _biases.begin());
}
