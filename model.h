#include <iostream>

struct GPT2Config {
    int block_size = 1024;
    int vocab_size = 50257;
    int n_layer = 12;
    int n_head = 12;
    int n_embd = 768;
};