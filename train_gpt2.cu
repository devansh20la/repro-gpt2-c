#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>

#include "tensor.h"
#include "model/linear.h"
#include "model/embedding.h"

using namespace std;

// Data class with config params
struct TrainConfig {
    int batch_size = 24;
    int block_size = 1024;
    float batch_size_in_token_req = static_cast<float>(1 << 19);  // ~0.5M tokens
    int train_steps = 1000;
    float lr = 6e-4f;
    tuple<float, float> betas{0.9f, 0.95f};
    float eps = 1e-8f;
    int max_steps = 100;
    float weighted_decay = 0.1f;
    int grad_accum_steps = 0;
    float min_lr = 0.0f;

    template <typename T>
    void print(T config_param, string config_name = "") const {
        cout << config_name << " " << config_param << endl;
    }

    void print_all_configs() const {
        print(batch_size, "batch_size");
        print(block_size, "block_size");
        print(batch_size_in_token_req, "batch_size_in_token_req");
        print(train_steps, "train_steps");
        print(lr, "lr");
        print(get<0>(betas), "betas[0]");
        print(get<1>(betas), "betas[1]");
        print(eps, "eps");
        print(max_steps, "max_steps");
        print(weighted_decay, "weighted_decay");
        print(grad_accum_steps, "grad_accum_steps");
        print(min_lr, "min_lr");
    }
};

class Dataset {
    private:
        int _block_size;
        int _batch_size;

    public:
        string _filepath;
        vector<uint16_t> inputData;
        int _num_batches;
        
        Dataset(const string& path, int block_size, int batch_size) : _filepath(path), _block_size(block_size), _batch_size(batch_size) {
            load(path);
            _num_batches = static_cast<int>(std::floor(inputData.size() / (_block_size * _batch_size)));
            cout << "Loaded " << inputData.size() << " tokens into dataset" << endl;
            cout << "Number of batches: " << _num_batches << endl;
        }

        void load(const string& path) {
            ifstream file(path, ios::binary | ios::ate);

            if (!file) {
                throw runtime_error("Dataset: failed to open " + path);
            };

            auto size = static_cast<size_t>(file.tellg());
            file.seekg(0, ios::beg);
            
            inputData.resize(size / sizeof(uint16_t));
            
            file.read(reinterpret_cast<char*>(inputData.data()), size);
            file.close();
        };

        // return a batch of data
        tuple<span<const uint16_t>, span<const uint16_t>> get_batch(int idx) {
            const uint16_t start_idx = static_cast<uint16_t>(idx * _block_size * _batch_size);
            
            if (idx >= _num_batches) {
                throw runtime_error("Dataset: batch out of range");
            }

            span<const uint16_t> x(inputData.data() + start_idx, _block_size * _batch_size);
            span<const uint16_t> y(inputData.data() + start_idx + 1, _block_size * _batch_size);
            return make_tuple(x, y);
        };
};

int main() {
    TrainConfig config;
    config.print_all_configs();
    printf("--------------------------------\n");
    printf("Loading train dataset...\n");
    const string train_file_path = "data/train.bin";
    Dataset train_dataset(train_file_path, config.block_size, config.batch_size);

    auto [x, y] = train_dataset.get_batch(0);
    (void)x;
    (void)y;

    // printf("--------------------------------\n");
    // Embedding embedding(10, 3);

    // thrust::device_vector<float> x_device(10);
    // for (int i = 0; i < 10; i++) {
    //     x_device[i] = i;
    // }

    // thrust::device_vector<float> y_device(10, 3);
    // embedding.forward(x_device, y_device);

    // for (int i = 0; i < 10; i++) {
    //     printf("%f ", static_cast<float>(y_device[i]));
    // }
    // printf("\n");

    printf("--------------------------------\n");
    // Before: thrust::device_vector<float> x_device(2 * 768, 1.0f) — flat, no shape
    // Now:    Tensor with shape [2, 768] — batch_size=2, in_features=768
    Tensor x_device({2, 768}, 1.0f);
    Tensor y_device;

    LinearLayer linear_layer(768, 1024);
    // Before: linear_layer.forward(x_device, 2, y_device) — had to pass batch_size=2
    // Now:    batch_size is read from x_device.shape(0) automatically
    linear_layer.forward(x_device, y_device);

    // After forward, y_device.shape() is [2, 1024]
    for (int i = 0; i < static_cast<int>(y_device.size()); i++) {
        printf("%f ", static_cast<float>(y_device.storage()[i]));
    }
    printf("\n");
    printf("--------------------------------\n");
    printf("Forward pass completed\n");
    
    return 0;
};
