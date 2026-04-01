#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

// Templated GPU tensor: a thrust device buffer + shape metadata.
//
// TensorBase<float>    -> activations, weights    (like torch.FloatTensor)
// TensorBase<uint16_t> -> token IDs               (like torch.LongTensor)
//
// The "Tensor" alias below defaults to float so existing code is unchanged.

template<typename T>
class TensorBase {
private:
    // Data buffer on GPU, we only store the data on GPU.
    // and it's shape. 
    // Note that shape is arbitrary because the data is stored in 
    // linear memory.
    thrust::device_vector<T> _data;
    std::vector<int> _shape;

public:
    TensorBase() = default;

    // Constructor to initialize the tensor with a given shape.
    explicit TensorBase(const std::vector<int>& shape)
        : _shape(shape) {
        // Calculate the total number of elements in the tensor
        // and resize the data buffer accordingly.
        size_t n = 1;
        for (int d : shape) n *= d;
        _data.resize(n);
    }

    // Constructor to initialize the tensor with a given shape and value.
    TensorBase(const std::vector<int>& shape, T value)
        : _shape(shape) {
        size_t n = 1;
        for (int d : shape) n *= d;
        _data.resize(n, value);
    }

    // Get the shape of the tensor
    const std::vector<int>& shape() const { return _shape; }

    // Supports negative indexing: shape(-1) == shape(ndim-1)
    int shape(int dim) const {
        if (dim < 0) dim += _shape.size();
        return _shape[dim];
    }

    int ndim() const { return _shape.size(); }
    size_t size() const { return _data.size(); }

    void resize(const std::vector<int>& new_shape) {
        size_t n = 1;
        for (int d : new_shape) n *= d;
        _data.resize(n);
        _shape = new_shape;
    }

    void reshape(const std::vector<int>& new_shape) {
        // Change shape metadata WITHOUT reallocating the GPU buffer.
        // The total number of elements must stay the same.
        //
        // This is free — no GPU work at all, just updating the host-side
        // shape vector.  Use it to reinterpret [B, T, C] as [B*T, C] etc.
    }

    // Get the data pointer of the tensor
    // this helps in passing the data to the cuda kernels.
    T* data_ptr() { return thrust::raw_pointer_cast(_data.data()); }
    const T* data_ptr() const { return thrust::raw_pointer_cast(_data.data()); }

    // Get the storage of the tensor
    // this helps in getting the data buffer on GPU.
    thrust::device_vector<T>& storage() { return _data; }
    const thrust::device_vector<T>& storage() const { return _data; }

    std::vector<TensorBase<T>> split(int chunks, int dim) const {
        // Function to  split into `chunks` equal pieces along `dim`.
        //
        // This will be helpful in :
        //   qkv has shape [B, T, 3*C]
        //   auto parts = qkv.split(3, -1);
        //   // parts[0] = [B, T, C]  (Q)
        //   // parts[1] = [B, T, C]  (K)
        //   // parts[2] = [B, T, C]  (V)

        return result;
    }
};

// Default alias — everywhere you wrote "Tensor" still means float.
using Tensor = TensorBase<float>;
