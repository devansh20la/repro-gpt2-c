#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

#include "gpt2_config.h"
#include "tensor.h"
#include "model.h"

using namespace std;

// Data class with config params
struct TrainConfig {
    int batch_size = 4;
    int block_size = 1024;
    int steps = 12;

    template <typename T>
    void print(T config_param, string config_name = "") const {
        cout << config_name << " " << config_param << endl;
    }

    void print_all_configs() const {
        print(batch_size, "batch_size");
        print(block_size, "block_size");
        print(steps, "steps");
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

    GPT2Config gpt2_config;
    gpt2_config.print_all_configs();

    printf("--------------------------------\n");
    printf("Loading train dataset...\n");
    const string train_file_path = "data/train.bin";
    Dataset train_dataset(train_file_path, config.block_size, config.batch_size);

    printf("--------------------------------\n");
    printf("Loading GPT2 model...\n");
    GPT2 gpt2(gpt2_config);
    gpt2.load_weights("checkpoints/weights.bin");

    printf("--------------------------------\n");
    printf("Running Forward passes over training dataset...\n");
    Tensor output({config.batch_size, config.block_size, gpt2_config.vocab_size});
    TensorBase<uint16_t> input_ids({config.batch_size, config.block_size});
    int batch_idx = 0;
    for (int i = 0; i < config.steps; i++) {
        batch_idx = i % train_dataset._num_batches;
        auto [x, y] = train_dataset.get_batch(batch_idx);

        // Copy input data to GPU
        {
            thrust::host_vector<uint16_t> hx(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                hx[i] = x[i];
            }
            thrust::copy(
                hx.begin(), 
                hx.end(),
                thrust::device_ptr<uint16_t>(input_ids.data_ptr())
            );
        }
        gpt2.forward(input_ids, output);
    }
    printf("--------------------------------\n");
    printf("Forward passes completed\n");



    // for (int i = 0; i < config.train_steps; i++) {
    //     auto batch_idx = i % 12;
    //     auto [x, y] = train_dataset.get_batch(batch_idx);
    //     TensorBase<uint16_t> input_ids({config.batch_size, config.block_size});
    //     {
    //         thrust::host_vector<uint16_t> hx(x.size());
    //         for (size_t i = 0; i < x.size(); ++i) {
    //             hx[i] = x[i];
    //         }
    //         thrust::copy(
    //             hx.begin(),
    //             hx.end(),
    //             thrust::device_ptr<uint16_t>(input_ids.data_ptr()));
    //     }
    //     gpt2.forward(input_ids, output);
    //     printf("Batch %d / %d | Train Step %d / %d Completed \n", batch_idx + 1, config.batch_size, i + 1, config.train_steps);
    // }
    // printf("Output: ");
    // printf("Shape: ");
    // for (int i = 0; i < static_cast<int>(output.shape().size()); i++) {
    //     printf("%d \n", output.shape()[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < static_cast<int>(output.size()); i++) {
    //     printf("%f ", static_cast<float>(output.storage()[i]));
    // }
    // printf("\n");

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

    // printf("--------------------------------\n");
    // // Before: thrust::device_vector<float> x_device(2 * 768, 1.0f) — flat, no shape
    // // Now:    Tensor with shape [2, 768] — batch_size=2, in_features=768
    // Tensor x_device({2, 768}, 1.0f);
    // Tensor y_device;

    // LinearLayer linear_layer(768, 1024);
    // // Before: linear_layer.forward(x_device, 2, y_device) — had to pass batch_size=2
    // // Now:    batch_size is read from x_device.shape(0) automatically
    // linear_layer.forward(x_device, y_device);

    // // After forward, y_device.shape() is [2, 1024]
    // for (int i = 0; i < static_cast<int>(y_device.size()); i++) {
    //     printf("%f ", static_cast<float>(y_device.storage()[i]));
    // }
    // printf("\n");
    // printf("--------------------------------\n");
    // printf("Forward pass completed\n");
    
    return 0;
};
