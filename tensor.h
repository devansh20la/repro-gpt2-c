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
    thrust::device_vector<T> _data;
    std::vector<int> _shape;

public:
    TensorBase() = default;

    explicit TensorBase(const std::vector<int>& shape)
        : _shape(shape) {
        size_t n = 1;
        for (int d : shape) n *= d;
        _data.resize(n);
    }

    TensorBase(const std::vector<int>& shape, T value)
        : _shape(shape) {
        size_t n = 1;
        for (int d : shape) n *= d;
        _data.resize(n, value);
    }

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

    // Change shape metadata WITHOUT reallocating the GPU buffer.
    // The total number of elements must stay the same.
    //
    // PyTorch equivalent:  t.reshape(new_shape)  or  t.view(new_shape)
    //
    // This is free — no GPU work at all, just updating the host-side
    // shape vector.  Use it to reinterpret [B, T, C] as [B*T, C] etc.
    void reshape(const std::vector<int>& new_shape) {
        size_t n = 1;
        for (int d : new_shape) n *= d;
        if (n != _data.size()) {
            throw std::invalid_argument(
                "TensorBase::reshape: new total size (" + std::to_string(n) +
                ") differs from current (" + std::to_string(_data.size()) + ")");
        }
        _shape = new_shape;
    }

    T* data_ptr() { return thrust::raw_pointer_cast(_data.data()); }
    const T* data_ptr() const { return thrust::raw_pointer_cast(_data.data()); }

    thrust::device_vector<T>& storage() { return _data; }
    const thrust::device_vector<T>& storage() const { return _data; }

    // Split into `chunks` equal pieces along `dim`.
    //
    // PyTorch equivalent:
    //   q, k, v = qkv.chunk(3, dim=-1)
    //
    // Example:
    //   qkv has shape [B, T, 3*C]
    //   auto parts = qkv.split(3, -1);
    //   // parts[0] = [B, T, C]  (Q)
    //   // parts[1] = [B, T, C]  (K)
    //   // parts[2] = [B, T, C]  (V)
    //
    // This performs a GPU-to-GPU copy (not a view) using cudaMemcpy2D
    // to handle the strided memory layout.
    std::vector<TensorBase<T>> split(int chunks, int dim) const {
        if (dim < 0) dim += ndim();
        if (dim < 0 || dim >= ndim()) {
            throw std::out_of_range("Tensor::split: dim out of range");
        }
        if (_shape[dim] % chunks != 0) {
            throw std::invalid_argument(
                "Tensor::split: dimension " + std::to_string(_shape[dim]) +
                " not divisible by " + std::to_string(chunks));
        }

        const int chunk_size = _shape[dim] / chunks;

        // Build the output shape: same as input but with dim shrunk.
        // e.g. [B, T, 3C] with chunks=3 on dim=2 -> [B, T, C]
        std::vector<int> out_shape = _shape;
        out_shape[dim] = chunk_size;

        // Compute strides to figure out the memory layout.
        // "inner" = product of dims after `dim` (the contiguous part within each row)
        // "outer" = product of dims before `dim` (number of "rows")
        // "full_row" = _shape[dim] * inner (full row width in the source)
        size_t inner = 1;
        for (int i = dim + 1; i < ndim(); i++) inner *= _shape[i];
        size_t outer = 1;
        for (int i = 0; i < dim; i++) outer *= _shape[i];

        const size_t full_row_elems = static_cast<size_t>(_shape[dim]) * inner;
        const size_t chunk_row_elems = static_cast<size_t>(chunk_size) * inner;

        std::vector<TensorBase<T>> result;
        result.reserve(chunks);

        for (int c = 0; c < chunks; c++) {
            TensorBase<T> out(out_shape);

            // cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)
            //
            // Think of it as copying a rectangle out of a grid:
            //   src row width (spitch) = full_row_elems (the entire [3C * inner] row)
            //   dst row width (dpitch) = chunk_row_elems (just [C * inner])
            //   width         = chunk_row_elems * sizeof(T) (bytes to copy per row)
            //   height        = outer (number of rows)
            //   src start     = data + c * chunk_row_elems (offset to this chunk)
            cudaMemcpy2D(
                out.data_ptr(),
                chunk_row_elems * sizeof(T),
                data_ptr() + c * chunk_row_elems,
                full_row_elems * sizeof(T),
                chunk_row_elems * sizeof(T),
                outer,
                cudaMemcpyDeviceToDevice);

            result.push_back(std::move(out));
        }

        return result;
    }
};

// Default alias — everywhere you wrote "Tensor" still means float.
using Tensor = TensorBase<float>;
