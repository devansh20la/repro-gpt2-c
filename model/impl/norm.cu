

LayerNorm::LayerNorm(int n_embd) : _n_embd(n_embd) {
    _weights.resize({n_embd});
    _biases.resize({n_embd});

    init_weights(0.0f, 1.0f);
} 

void LayerNorm::forward(const Tensor& input, Tensor& output) {
    input_shape = input.shape();

    if (input_shape.size() != 3) {
        throw std::invalid_argument("LayerNorm::forward: input must have 3 dimensions");
    }

    // resize output and weights and biases
    // input shape: [batch_size, sequence_length, n_embd]
    // output shape: [batch_size, sequence_length, n_embd]
    // weights shape: [n_embd] : means
    // biases shape: [n_embd] : variances
    output.resize({input_shape[0], input_shape[1], input_shape[2]});
    _weights.resize({input_shape[2]});
    _biases.resize({input_shape[2]});

    

