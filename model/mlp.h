#pragma once

#include "../tensor.h"
#include "linear.h"
#include "act.h"
#include <string>
#include <unordered_map>
#include <vector>

class MLP {
    private:
        int _in_features;
        int _scaling_factor;
        LinearLayer _fc1;
        GELU _gelu;
        LinearLayer _fc2;
    public:
        MLP(int in_features, int scaling_factor);
        void forward(const Tensor& input, Tensor& output);
        void init_weights(float mean, float std);
        void load_weights(
            const std::unordered_map<std::string, std::vector<float>>& tensors,
            const std::string& prefix);
};