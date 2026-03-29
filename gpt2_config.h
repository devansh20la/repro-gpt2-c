#pragma once

#include <iostream>
#include <string>
#include "tensor.h"

struct GPT2Config {
    int block_size = 1024;
    int vocab_size = 50257;
    int n_layers = 12;
    int n_head = 12;
    int n_embd = 768;
    /// MLP inner dim = n_embd * scaling_factor (GPT-2 uses 4).
    int scaling_factor = 4;

    template <typename T>
    void print(T config_param, std::string config_name = "") const {
        std::cout << config_name << " " << config_param << std::endl;
    }

    void print_all_configs() const {
        print(block_size, "block_size");
        print(vocab_size, "vocab_size");
        print(n_layers, "n_layers");
        print(n_head, "n_head");
        print(n_embd, "n_embd");
        print(scaling_factor, "scaling_factor");
    }
};
