#include <iostream>
#include <string>
#include <tuple>

using namespace std;
{
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
            string _filepath;
            vector<uint16_t> _tokens;

        public:
            Dataset(string path) = default;
            void load(string path) {
                ifstream file(path, ios::binary | ios::ate);
                if (!file) {
                    throw runtime_error("Dataset: failed to open " + path);
                };

                auto size = static_cast<size_t>(file.tellg());
                _tokens.resize(size / sizeof(uint16_t));
                file.read(, size);
                file.close();
                _filepath = path;`
            };
    };
    int main() {
        TrainConfig config;
        config.print_all_configs();
        return 0;
        // # load dataset likely shakespear 

        // # load model 

        // # intialize optimizer

        // # train model


    }
}