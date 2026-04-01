#pragma once

#include "../tensor.h"
#include "linear.h"
#include "act.h"
#include <string>
#include <unordered_map>
#include <vector>

// PyTorch: two Linear layers with GELU — `c_fc` expands by `scaling_factor`, `c_proj` projects back.
// Forward: x -> fc1 -> GELU -> fc2 (same as GPT-2 MLP block).
// This is nothing but multiple linear layers with a GELU activation function in between.
// You should implement the Linear and Gelu layers before implementing this.
class MLP {
private:
    int _in_features;
    int _scaling_factor;
    LinearLayer _fc1;
    GELU _gelu;
    LinearLayer _fc2;

public:
    // Constructor to initialize the MLP layer
    MLP(int in_features, int scaling_factor);

    // forward pass the input through the MLP layer
    void forward(const Tensor& input, Tensor& output);

    // initialize the weights of the MLP layer
    void init_weights(float mean, float std);

    // load the weights of the MLP layer from a file
    // Note that the weights of the MLP layer are stored in a file in the same format as the weights of the linear layers.
    // You should implement the load_weights function in the Linear layer before implementing this.
    void load_weights(
        const std::unordered_map<std::string, std::vector<float>>& tensors,
        const std::string& prefix); 
};
