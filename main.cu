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
struct Config {
    int batch_size = 24;
    int block_size = 1024;

    template <typename T>
    void print(T config_param, string config_name = "") const {
        cout << config_name << " " << config_param << endl;
    }

    void print_all_configs() const {
        print(batch_size, "batch_size");
        print(block_size, "block_size");
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
        
        Dataset(const string& path, int block_size, int batch_size) : 
            _filepath(path), 
            _block_size(block_size),
            _batch_size(batch_size),
            inputData(0), 
            _num_batches(0) {
                load(_filepath);
                _num_batches = static_cast<int>(std::floor(inputData.size() / (_block_size * _batch_size)));
                printf("Loaded %d tokens into dataset\n", inputData.size());
                printf("Number of batches: %d\n", _num_batches);
            }

        void load(const string& path) {
            // Open the dataset binary file 
            // Check if the file is opened successfully
            // Get the size of the file
            // Read the file into the inputData vector
            // Close the file
        };

        // return a batch of data
        tuple<span<const uint16_t>, span<const uint16_t>> get_batch(int idx) {
            // Function to get a batch of data from the dataset
            // 1. Calculate the start index of the batch
            // 2. Check if the batch index is out of range
            // 3. Create spans for the input and output data. Note that label y is just inputx shifted by 1.
            // 4. Return the spans
        };
};

int main() {
    // Initialize the Config for forward pass
    Config config;

    // Initialize the GPT2 configuration
    GPT2Config gpt2_config;

    // Initialize the GPT2 model
    GPT2 gpt2(gpt2_config);

    // Initialize the Dataset for training
    const string train_file_path = "data/train.bin";
    Dataset train_dataset(train_file_path, config.block_size, config.batch_size);

    // Get a batch of data from the dataset
    auto [x, y] = train_dataset.get_batch(0);

    // Forward pass the data through the model

    
    return 0;
};
