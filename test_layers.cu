// ============================================================================
//  test_layers.cu — unit tests for every layer in model/
//
//  Build:
//    nvcc -O2 -o test_layers \
//      test_layers.cu model/impl/act.cu model/impl/embedding.cu \
//      model/impl/linear.cu model/impl/causal_attn.cu model/impl/norm.cu
//
//  Run:
//    ./test_layers
//
//  Each test creates known inputs on the GPU, runs the layer's forward pass,
//  copies the output back to the CPU, and compares against hand-computed
//  expected values.  Float comparisons use an absolute tolerance (default 1e-4)
//  because GPU floating-point arithmetic can differ from the host by small
//  amounts due to fused multiply-add (FMA) and different rounding modes.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "tensor.h"
#include "model/act.h"
#include "model/embedding.h"
#include "model/linear.h"
#include "model/causal_attn.h"
#include "model/norm.h"

#include <thrust/copy.h>
#include <thrust/host_vector.h>

// ─── Test bookkeeping ───────────────────────────────────────────────────────

static int g_passed = 0;
static int g_failed = 0;

// Copy a GPU Tensor to a host std::vector so we can inspect element values
// on the CPU.  thrust::host_vector handles the cudaMemcpy for us.
static std::vector<float> to_host(const Tensor& t) {
    thrust::host_vector<float> h = t.storage();
    return std::vector<float>(h.begin(), h.end());
}

static bool close_enough(float a, float b, float atol = 1e-4f) {
    return std::fabs(a - b) <= atol;
}

// Element-wise comparison of a GPU tensor against expected host values.
static bool check_tensor(const Tensor& gpu,
                          const std::vector<float>& expected,
                          float atol = 1e-4f) {
    auto actual = to_host(gpu);
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); i++) {
        if (!close_enough(actual[i], expected[i], atol)) return false;
    }
    return true;
}

static void report(const char* name, bool passed) {
    if (passed) {
        printf("  [PASS] %s\n", name);
        g_passed++;
    } else {
        printf("  [FAIL] %s\n", name);
        g_failed++;
    }
}

// Helper: upload a host float vector into an existing device Tensor.
static void upload(Tensor& t, const std::vector<float>& h) {
    thrust::copy(h.begin(), h.end(), t.storage().begin());
}

// Helper: upload a host uint16 vector into an existing device TensorBase.
static void upload_u16(TensorBase<uint16_t>& t,
                       const std::vector<uint16_t>& h) {
    thrust::copy(h.begin(), h.end(), t.storage().begin());
}

// ═══════════════════════════════════════════════════════════════════════════
//  1. Tensor (tensor.h) — construction, shape helpers, split
// ═══════════════════════════════════════════════════════════════════════════

static void test_tensor_construction() {
    //  Tensor({2, 3}, 1.0f) should create a 2×3 buffer filled with 1.0.
    Tensor t({2, 3}, 1.0f);

    bool ok = true;
    ok &= (t.ndim() == 2);
    ok &= (t.shape(0) == 2);
    ok &= (t.shape(1) == 3);
    ok &= (t.size() == 6);

    auto v = to_host(t);
    for (float x : v) ok &= close_enough(x, 1.0f);

    report("Tensor construction [2,3] filled with 1.0", ok);
}

static void test_tensor_negative_indexing() {
    //  shape(-1) should return the last dimension, shape(-2) the second-last, etc.
    Tensor t({4, 5, 6});
    bool ok = (t.shape(-1) == 6) && (t.shape(-2) == 5) && (t.shape(-3) == 4);
    report("Tensor negative dim indexing", ok);
}

static void test_tensor_resize() {
    //  resize() changes both the shape metadata AND reallocates the buffer.
    Tensor t({2, 3}, 0.0f);
    t.resize({4, 5});
    bool ok = (t.ndim() == 2) && (t.shape(0) == 4) && (t.shape(1) == 5)
           && (t.size() == 20);
    report("Tensor resize [2,3] -> [4,5]", ok);
}

static void test_tensor_split_2way() {
    //  A [2, 6] tensor:
    //    row0: [0  1  2 | 3  4  5]
    //    row1: [6  7  8 | 9 10 11]
    //
    //  split(2, dim=1) → two [2, 3] tensors:
    //    chunk0: [[0,1,2], [6,7,8]]
    //    chunk1: [[3,4,5], [9,10,11]]
    //
    //  cudaMemcpy2D copies a non-contiguous "rectangle" out of device memory.
    Tensor t({2, 6});
    {
        std::vector<float> h(12);
        for (int i = 0; i < 12; i++) h[i] = static_cast<float>(i);
        upload(t, h);
    }

    auto parts = t.split(2, 1);

    bool ok = (parts.size() == 2);
    ok &= (parts[0].shape(0) == 2 && parts[0].shape(1) == 3);
    ok &= (parts[1].shape(0) == 2 && parts[1].shape(1) == 3);
    ok &= check_tensor(parts[0], {0, 1, 2, 6, 7, 8});
    ok &= check_tensor(parts[1], {3, 4, 5, 9, 10, 11});

    report("Tensor::split [2,6] -> 2x[2,3] along dim=1", ok);
}

static void test_tensor_split_3way() {
    //  Simulates the QKV split used in attention:
    //    [1, 2, 6] → 3 chunks along dim=-1 → 3x [1, 2, 2]
    //
    //  Data layout:
    //    [[[0,1, 2,3, 4,5],
    //      [6,7, 8,9, 10,11]]]
    //
    //  chunk0 (Q): [[[0,1], [6,7]]]
    //  chunk1 (K): [[[2,3], [8,9]]]
    //  chunk2 (V): [[[4,5], [10,11]]]
    Tensor t({1, 2, 6});
    {
        std::vector<float> h(12);
        for (int i = 0; i < 12; i++) h[i] = static_cast<float>(i);
        upload(t, h);
    }

    auto parts = t.split(3, -1);   // -1 means last dim

    bool ok = (parts.size() == 3);
    for (auto& p : parts) {
        ok &= (p.shape(0) == 1 && p.shape(1) == 2 && p.shape(2) == 2);
    }
    ok &= check_tensor(parts[0], {0, 1, 6, 7});
    ok &= check_tensor(parts[1], {2, 3, 8, 9});
    ok &= check_tensor(parts[2], {4, 5, 10, 11});

    report("Tensor::split [1,2,6] -> 3x[1,2,2] (QKV style)", ok);
}

static void test_tensor_reshape() {
    //  reshape() changes the shape metadata without touching GPU memory.
    //  The total element count must stay the same.
    //    [2, 6] (12 elements) → [3, 4] (12 elements): OK
    //    [2, 6] (12 elements) → [5, 5] (25 elements): throws
    Tensor t({2, 6}, 1.0f);
    t.reshape({3, 4});

    bool ok = (t.ndim() == 2) && (t.shape(0) == 3) && (t.shape(1) == 4)
           && (t.size() == 12);

    // Verify reshape with wrong total size throws
    bool threw = false;
    try { t.reshape({5, 5}); }
    catch (const std::invalid_argument&) { threw = true; }
    ok &= threw;

    report("Tensor::reshape metadata-only + reject size mismatch", ok);
}

// ═══════════════════════════════════════════════════════════════════════════
//  2. ReLU (model/act.h) — element-wise max(x, 0)
// ═══════════════════════════════════════════════════════════════════════════

static void test_relu_basic() {
    //  ReLU zeroes out negatives and keeps positives untouched.
    //    input:    [-2, -1, 0, 1, 2]
    //    expected: [ 0,  0, 0, 1, 2]
    Tensor input({1, 5});
    upload(input, {-2, -1, 0, 1, 2});

    ReLU relu;
    Tensor output;
    relu.forward(input, output);

    report("ReLU basic [-2,-1,0,1,2]",
           check_tensor(output, {0, 0, 0, 1, 2}));
}

static void test_relu_all_negative() {
    //  Every element is negative → entire output should be zeros.
    Tensor input({1, 4});
    upload(input, {-5, -3, -1, -0.001f});

    ReLU relu;
    Tensor output;
    relu.forward(input, output);

    report("ReLU all-negative -> all zeros",
           check_tensor(output, {0, 0, 0, 0}));
}

static void test_relu_all_positive() {
    //  All positive input → output unchanged (pass-through).
    Tensor input({1, 3});
    upload(input, {0.5f, 1.0f, 100.0f});

    ReLU relu;
    Tensor output;
    relu.forward(input, output);

    report("ReLU all-positive (pass-through)",
           check_tensor(output, {0.5f, 1.0f, 100.0f}));
}

static void test_relu_preserves_shape() {
    //  Output shape must match input shape.
    Tensor input({3, 4}, 1.0f);
    ReLU relu;
    Tensor output;
    relu.forward(input, output);

    bool ok = (output.ndim() == 2)
           && (output.shape(0) == 3)
           && (output.shape(1) == 4);
    report("ReLU preserves shape [3,4]", ok);
}

// ═══════════════════════════════════════════════════════════════════════════
//  3. GELU (model/act.h) — smooth activation used by GPT-2
//
//  GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
//  Key properties:
//    - Large positive x → output ≈ x  (like ReLU)
//    - Large negative x → output ≈ 0  (like ReLU)
//    - Around 0 → smooth curve       (unlike ReLU's sharp corner)
//    - GELU(0) = 0  exactly
// ═══════════════════════════════════════════════════════════════════════════

// Helper: compute GELU on the host for expected values.
static float host_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void test_gelu_basic() {
    //  Test with a mix of negative, zero, and positive values.
    //  Compare against the same formula computed on the CPU.
    Tensor input({1, 5});
    upload(input, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});

    GELU gelu;
    Tensor output;
    gelu.forward(input, output);

    std::vector<float> expected;
    for (float x : {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}) {
        expected.push_back(host_gelu(x));
    }

    report("GELU basic [-2,-1,0,1,2]", check_tensor(output, expected));
}

static void test_gelu_zero() {
    //  GELU(0) = 0.5 * 0 * (...) = 0 exactly.
    Tensor input({1, 1}, 0.0f);

    GELU gelu;
    Tensor output;
    gelu.forward(input, output);

    report("GELU(0) == 0", check_tensor(output, {0.0f}));
}

static void test_gelu_large_positive() {
    //  For large positive x, GELU(x) ≈ x because tanh → 1, gate → 1.
    Tensor input({1, 2});
    upload(input, {5.0f, 10.0f});

    GELU gelu;
    Tensor output;
    gelu.forward(input, output);

    auto vals = to_host(output);
    bool ok = close_enough(vals[0], 5.0f, 1e-3f)
           && close_enough(vals[1], 10.0f, 1e-3f);

    report("GELU large positive ≈ identity", ok);
}

static void test_gelu_large_negative() {
    //  For large negative x, GELU(x) ≈ 0 because tanh → -1, gate → 0.
    Tensor input({1, 2});
    upload(input, {-5.0f, -10.0f});

    GELU gelu;
    Tensor output;
    gelu.forward(input, output);

    auto vals = to_host(output);
    bool ok = close_enough(vals[0], 0.0f, 1e-3f)
           && close_enough(vals[1], 0.0f, 1e-3f);

    report("GELU large negative ≈ 0", ok);
}

static void test_gelu_preserves_shape() {
    Tensor input({3, 4}, 1.0f);
    GELU gelu;
    Tensor output;
    gelu.forward(input, output);

    bool ok = (output.ndim() == 2)
           && (output.shape(0) == 3)
           && (output.shape(1) == 4);
    report("GELU preserves shape [3,4]", ok);
}

// ═══════════════════════════════════════════════════════════════════════════
//  4. Softmax (model/act.h) — row-wise softmax along last dimension
//
//  For a row [x0, x1, ..., xN]:
//    softmax(xi) = exp(xi - max) / Σ exp(xj - max)
//
//  Subtracting max prevents exp() from overflowing (numerical stability).
// ═══════════════════════════════════════════════════════════════════════════

static void test_softmax_basic() {
    //  softmax([1, 2, 3]):
    //    e^1 / (e^1 + e^2 + e^3) ≈ 0.0900
    //    e^2 / (e^1 + e^2 + e^3) ≈ 0.2447
    //    e^3 / (e^1 + e^2 + e^3) ≈ 0.6652
    Tensor input({1, 3});
    upload(input, {1, 2, 3});

    Softmax sm;
    Tensor output;
    sm.forward(input, output);

    float e1 = expf(1), e2 = expf(2), e3 = expf(3);
    float s = e1 + e2 + e3;

    report("Softmax [1,2,3]",
           check_tensor(output, {e1/s, e2/s, e3/s}));
}

static void test_softmax_rows_independent() {
    //  Two rows are softmax'd independently:
    //    row0: [1, 2, 3]  →  [0.0900, 0.2447, 0.6652]
    //    row1: [3, 2, 1]  →  [0.6652, 0.2447, 0.0900]   (mirror of row0)
    Tensor input({2, 3});
    upload(input, {1, 2, 3, 3, 2, 1});

    Softmax sm;
    Tensor output;
    sm.forward(input, output);

    float e1 = expf(1), e2 = expf(2), e3 = expf(3);
    float s = e1 + e2 + e3;

    report("Softmax 2 independent rows",
           check_tensor(output, {e1/s, e2/s, e3/s,
                                 e3/s, e2/s, e1/s}));
}

static void test_softmax_sums_to_one() {
    //  The defining property: each row's probabilities sum to exactly 1.0.
    //  We test with three very different rows (big, small, negative values).
    Tensor input({3, 5});
    upload(input, {
        1, 5, 2, 4, 3,
        10, 20, 30, 40, 50,
        -1, -2, -3, -4, -5
    });

    Softmax sm;
    Tensor output;
    sm.forward(input, output);

    auto vals = to_host(output);
    bool ok = true;
    for (int row = 0; row < 3; row++) {
        float row_sum = 0;
        for (int col = 0; col < 5; col++) {
            float v = vals[row * 5 + col];
            row_sum += v;
            ok &= (v >= 0.0f);        // softmax outputs are always non-negative
        }
        ok &= close_enough(row_sum, 1.0f, 1e-4f);
    }

    report("Softmax rows sum to 1.0", ok);
}

static void test_softmax_uniform_input() {
    //  If every element in a row is identical, softmax produces a uniform
    //  distribution: 1/N for each element.
    //    softmax([5, 5, 5, 5]) = [0.25, 0.25, 0.25, 0.25]
    Tensor input({1, 4}, 5.0f);

    Softmax sm;
    Tensor output;
    sm.forward(input, output);

    report("Softmax uniform input -> 1/N each",
           check_tensor(output, {0.25f, 0.25f, 0.25f, 0.25f}));
}

static void test_softmax_3d_tensor() {
    //  3D input [B, T, C]: softmax still operates on the last dimension (C).
    //  shape [2, 1, 3] → two batches, each with one row of length 3.
    Tensor input({2, 1, 3});
    upload(input, {1, 2, 3, 3, 2, 1});

    Softmax sm;
    Tensor output;
    sm.forward(input, output);

    float e1 = expf(1), e2 = expf(2), e3 = expf(3);
    float s = e1 + e2 + e3;

    bool ok = check_tensor(output, {e1/s, e2/s, e3/s,
                                    e3/s, e2/s, e1/s});
    ok &= (output.shape(0) == 2 && output.shape(1) == 1 && output.shape(2) == 3);

    report("Softmax 3D [2,1,3] along last dim", ok);
}

// ═══════════════════════════════════════════════════════════════════════════
//  4. LinearLayer (model/linear.h) — y = x @ W^T + b
//
//  Weights are stored row-major as [out_features, in_features].
//  The kernel computes:  output[batch, j] = bias[j] + Σ_i input[batch, i] * W[j, i]
//  Biases are initialised to 0 by init_weights(), so all tests below have b=0.
// ═══════════════════════════════════════════════════════════════════════════

static void test_linear_identity() {
    //  2×2 identity weight matrix:
    //    W = [[1, 0],
    //         [0, 1]]    flat: [1, 0, 0, 1]
    //
    //  input  = [[3, 7]]
    //  output = [[3*1+7*0, 3*0+7*1]] = [[3, 7]]
    LinearLayer layer(2, 2);
    float w[] = {1, 0, 0, 1};
    layer.set_weights(w, 4);

    Tensor input({1, 2});
    upload(input, {3.0f, 7.0f});

    Tensor output;
    layer.forward(input, output);

    report("Linear 2x2 identity",
           check_tensor(output, {3.0f, 7.0f}));
}

static void test_linear_known_weights() {
    //  in=3, out=2
    //    W = [[1,  0, -1],     ← row 0 (output feature 0)
    //         [0,  1,  0]]     ← row 1 (output feature 1)
    //    flat: [1, 0, -1, 0, 1, 0]
    //
    //  input  = [[1, 2, 3]]
    //  out[0] = 1*1 + 2*0 + 3*(-1) = -2
    //  out[1] = 1*0 + 2*1 + 3*0    =  2
    LinearLayer layer(3, 2);
    float w[] = {1, 0, -1, 0, 1, 0};
    layer.set_weights(w, 6);

    Tensor input({1, 3});
    upload(input, {1.0f, 2.0f, 3.0f});

    Tensor output;
    layer.forward(input, output);

    bool ok = check_tensor(output, {-2.0f, 2.0f});
    ok &= (output.shape(0) == 1 && output.shape(1) == 2);

    report("Linear [1,3]->[1,2] known weights", ok);
}

static void test_linear_batched() {
    //  batch=3, in=2, out=2
    //    W = [[1, 2],      flat: [1, 2, 3, 4]
    //         [3, 4]]
    //
    //  For each row of the input we compute input @ W^T:
    //    [1, 0] → [1*1+0*2, 1*3+0*4] = [1, 3]
    //    [0, 1] → [0*1+1*2, 0*3+1*4] = [2, 4]
    //    [1, 1] → [1*1+1*2, 1*3+1*4] = [3, 7]
    LinearLayer layer(2, 2);
    float w[] = {1, 2, 3, 4};
    layer.set_weights(w, 4);

    Tensor input({3, 2});
    upload(input, {1, 0, 0, 1, 1, 1});

    Tensor output;
    layer.forward(input, output);

    bool ok = check_tensor(output, {1, 3, 2, 4, 3, 7});
    ok &= (output.shape(0) == 3 && output.shape(1) == 2);

    report("Linear batched [3,2]->[3,2]", ok);
}

static void test_linear_3d_input() {
    //  LinearLayer now supports ≥1D, matching PyTorch's nn.Linear.
    //  For 3D input [B, T, C_in], it treats B*T as the batch axis
    //  and produces output [B, T, C_out].
    //
    //  W = [[1, 0, -1],    flat: [1, 0, -1, 0, 1, 0]
    //       [0, 1,  0]]
    //
    //  input [2, 2, 3]:
    //    batch0, pos0: [1, 2, 3] → [1-3, 2] = [-2, 2]
    //    batch0, pos1: [0, 0, 0] → [ 0,  0] = [ 0, 0]
    //    batch1, pos0: [1, 0, 0] → [ 1,  0] = [ 1, 0]
    //    batch1, pos1: [0, 0, 1] → [-1,  0] = [-1, 0]
    LinearLayer layer(3, 2);
    float w[] = {1, 0, -1, 0, 1, 0};
    layer.set_weights(w, 6);

    Tensor input({2, 2, 3});
    upload(input, {1,2,3, 0,0,0, 1,0,0, 0,0,1});

    Tensor output;
    layer.forward(input, output);

    bool ok = check_tensor(output, {-2,2, 0,0, 1,0, -1,0});
    ok &= (output.ndim() == 3);
    ok &= (output.shape(0) == 2 && output.shape(1) == 2 && output.shape(2) == 2);

    report("Linear 3D [2,2,3]->[2,2,2]", ok);
}

static void test_linear_rejects_wrong_features() {
    //  input.shape(1) must equal in_features.
    //  Layer expects 4, but input has 5 → should throw.
    LinearLayer layer(4, 2);
    Tensor input({1, 5});
    Tensor output;

    bool threw = false;
    try {
        layer.forward(input, output);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    report("Linear rejects mismatched in_features", threw);
}

// ═══════════════════════════════════════════════════════════════════════════
//  5. Embedding (model/embedding.h) — lookup table: token id → float vector
//
//  The constructor calls init_embeddings(0, 1) which fills the table with
//  N(0,1) random values using seed 42.  We replicate the same RNG on the
//  host side to compute exact expected values.
// ═══════════════════════════════════════════════════════════════════════════

static void test_embedding_output_shape() {
    //  Embedding(vocab=100, dim=16)
    //    input  [2, 5]  (2 sequences of 5 tokens)
    //    output [2, 5, 16]
    Embedding emb(100, 16);

    TensorBase<uint16_t> input({2, 5});
    {
        std::vector<uint16_t> h = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        upload_u16(input, h);
    }

    Tensor output;
    emb.forward(input, output);

    bool ok = (output.ndim() == 3)
           && (output.shape(0) == 2)
           && (output.shape(1) == 5)
           && (output.shape(2) == 16);

    report("Embedding output shape [2,5]->[2,5,16]", ok);
}

static void test_embedding_same_token_same_vector() {
    //  Looking up the same token at different positions must return
    //  identical embedding vectors.
    Embedding emb(10, 4);

    TensorBase<uint16_t> input({1, 3});
    {
        std::vector<uint16_t> h = {3, 7, 3};  // token 3 appears at pos 0 and 2
        upload_u16(input, h);
    }

    Tensor output;
    emb.forward(input, output);

    auto vals = to_host(output);   // shape [1, 3, 4] → 12 floats
    bool ok = true;
    for (int i = 0; i < 4; i++) {
        ok &= close_enough(vals[0 * 4 + i], vals[2 * 4 + i]);
    }

    report("Embedding same token -> same vector", ok);
}

static void test_embedding_known_values() {
    //  Reproduce the exact random embedding table on the host:
    //    std::mt19937 rng(42);
    //    std::normal_distribution<float> dist(0, 1);
    //  Then verify that forward([0, 2, 4]) returns the right rows.
    const int vocab = 5, dim = 3;
    Embedding emb(vocab, dim);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> table(vocab * dim);
    for (int i = 0; i < vocab * dim; i++) table[i] = dist(rng);

    TensorBase<uint16_t> input({1, 3});
    {
        std::vector<uint16_t> h = {0, 2, 4};
        upload_u16(input, h);
    }

    Tensor output;
    emb.forward(input, output);

    auto vals = to_host(output);   // [1, 3, 3] → 9 floats
    bool ok = (vals.size() == 9);

    uint16_t tokens[] = {0, 2, 4};
    for (int t = 0; t < 3; t++) {
        for (int d = 0; d < dim; d++) {
            ok &= close_enough(vals[t * dim + d],
                               table[tokens[t] * dim + d]);
        }
    }

    report("Embedding matches seed-42 random table", ok);
}

static void test_embedding_out_of_range() {
    //  Token ids >= vocab_size are clamped to zero vectors by the kernel.
    Embedding emb(5, 3);

    TensorBase<uint16_t> input({1, 2});
    {
        std::vector<uint16_t> h = {0, 10};  // token 10 is out of range (vocab=5)
        upload_u16(input, h);
    }

    Tensor output;
    emb.forward(input, output);

    auto vals = to_host(output);   // [1, 2, 3] → 6 floats
    bool ok = true;
    for (int i = 3; i < 6; i++) {  // second token's embedding
        ok &= close_enough(vals[i], 0.0f);
    }

    report("Embedding out-of-range token -> zeros", ok);
}

// ═══════════════════════════════════════════════════════════════════════════
//  6. Attention (model/causal_attn.h) — scaled dot-product attention
//
//  scores  = Q @ K^T / √d_head   [B, H, T, T]
//  scores  = causal_mask(scores)  future positions → -∞
//  weights = softmax(scores)      each row sums to 1
//  output  = weights @ V          [B, H, T, d_head]
// ═══════════════════════════════════════════════════════════════════════════

static void test_attention_seq_len_1() {
    //  With seq_len=1 there's no masking (the single position [0,0] has
    //  col == row so it's kept).  The softmax of a single value is 1.0,
    //  so output = 1.0 * V = V.
    //
    //  Q = K = V with shape [B,H,T,d] = [1,1,1,2]
    //
    //  scores = Q·K^T / √2 = (1*1 + 2*2)/√2 = 5/√2 ≈ 3.535
    //  softmax([3.535]) = [1.0]
    //  output = 1.0 * V = [[1, 2]]
    Tensor q({1, 1, 1, 2}), k({1, 1, 1, 2}), v({1, 1, 1, 2});
    upload(q, {1, 2});
    upload(k, {1, 2});
    upload(v, {1, 2});

    Attention attn(2, 1);  // head_dim=2, n_heads=1
    Tensor output;
    attn.forward(q, k, v, output);

    bool ok = check_tensor(output, {1.0f, 2.0f});
    ok &= (output.shape(0) == 1 && output.shape(1) == 1 && output.shape(2) == 1
           && output.shape(3) == 2);

    report("Attention seq_len=1 (output == V)", ok);
}

static void test_attention_causal_mask() {
    //  Q = K = [[[1, 0],      shape [1, 2, 2]
    //            [0, 1]]]
    //  V = [[[1, 0],
    //        [0, 1]]]
    //
    //  Step 1: Q @ K^T / √2
    //    [0,0]: Q[0]·K[0] / √2 = 1/√2 ≈ 0.7071
    //    [0,1]: MASKED → -∞        (col=1 > row=0: future position)
    //    [1,0]: Q[1]·K[0] / √2 = 0/√2 = 0
    //    [1,1]: Q[1]·K[1] / √2 = 1/√2 ≈ 0.7071
    //
    //  Step 2: softmax (per row)
    //    row 0: softmax([0.7071, -∞]) = [1.0, 0.0]
    //    row 1: softmax([0, 0.7071])
    //      exp(0)      = 1.0
    //      exp(0.7071) ≈ 2.02811
    //      sum         ≈ 3.02811
    //      → [1/3.02811, 2.02811/3.02811] ≈ [0.33022, 0.66978]
    //
    //  Step 3: weights @ V
    //    row 0: [1.0*1+0.0*0, 1.0*0+0.0*1] = [1.0, 0.0]
    //    row 1: [0.33022*1+0.66978*0, 0.33022*0+0.66978*1] = [0.33022, 0.66978]
    Tensor q({1, 1, 2, 2}), k({1, 1, 2, 2}), v({1, 1, 2, 2});
    upload(q, {1, 0, 0, 1});
    upload(k, {1, 0, 0, 1});
    upload(v, {1, 0, 0, 1});

    Attention attn(2, 1);
    Tensor output;
    attn.forward(q, k, v, output);

    float s = 1.0f / sqrtf(2.0f);
    float exp_s = expf(s);
    float p0 = 1.0f / (1.0f + exp_s);
    float p1 = exp_s / (1.0f + exp_s);

    bool ok = check_tensor(output, {1.0f, 0.0f, p0, p1}, 1e-3f);
    ok &= (output.shape(0) == 1 && output.shape(1) == 1 && output.shape(2) == 2
           && output.shape(3) == 2);

    report("Attention causal mask (2x2)", ok);
}

static void test_attention_output_shape() {
    //  [B,H,T,d] → same shape
    Tensor q({2, 1, 4, 8}, 1.0f);
    Tensor k({2, 1, 4, 8}, 1.0f);
    Tensor v({2, 1, 4, 8}, 1.0f);

    Attention attn(8, 1);
    Tensor output;
    attn.forward(q, k, v, output);

    bool ok = (output.ndim() == 4)
           && (output.shape(0) == 2)
           && (output.shape(1) == 1)
           && (output.shape(2) == 4)
           && (output.shape(3) == 8);

    report("Attention output shape [2,4,8]", ok);
}

static void test_attention_first_row_copies_first_value() {
    //  The first position (row 0) can only attend to itself (everything
    //  else is masked).  So output[0] should always equal V[0], regardless
    //  of the query/key values.
    Tensor q({1, 1, 3, 4}, 0.0f);
    Tensor k({1, 1, 3, 4}, 0.0f);
    Tensor v({1, 1, 3, 4});
    upload(v, {
        10, 20, 30, 40,    // V[0] — this should appear in output[0]
         1,  2,  3,  4,
         5,  6,  7,  8
    });

    Attention attn(4, 1);
    Tensor output;
    attn.forward(q, k, v, output);

    auto vals = to_host(output);
    bool ok = true;
    ok &= close_enough(vals[0], 10.0f);
    ok &= close_enough(vals[1], 20.0f);
    ok &= close_enough(vals[2], 30.0f);
    ok &= close_enough(vals[3], 40.0f);

    report("Attention first row always == V[0]", ok);
}

// ═══════════════════════════════════════════════════════════════════════════
//  7. CausalMultiHeadedAttention (model/causal_attn.h) — full attention block
//
//  Linear projection [B, T, C] → [B, T, 3C] then split + Attention.
//  Hard to test exact values (random projection weights), so we test
//  shapes and basic properties.
// ═══════════════════════════════════════════════════════════════════════════

static void test_causal_attention_output_shape() {
    //  n_embd=8, n_heads=1 → head_dim=8
    //  input [2, 4, 8] → output [2, 4, 8]
    CausalMultiHeadedAttention cattn(8, 1);
    Tensor input({2, 4, 8}, 1.0f);
    Tensor output;
    cattn.forward(input, output);

    bool ok = (output.ndim() == 3)
           && (output.shape(0) == 2)
           && (output.shape(1) == 4)
           && (output.shape(2) == 8);

    report("CausalMultiHeadedAttention output shape [2,4,8]", ok);
}

static void test_causal_attention_deterministic() {
    //  Running the same input through the same layer twice must produce
    //  identical outputs (no randomness in the forward pass).
    CausalMultiHeadedAttention cattn(4, 1);
    Tensor input({1, 3, 4}, 1.0f);

    Tensor out1, out2;
    cattn.forward(input, out1);
    cattn.forward(input, out2);

    auto v1 = to_host(out1);
    auto v2 = to_host(out2);
    bool ok = (v1.size() == v2.size());
    for (size_t i = 0; i < v1.size(); i++) {
        ok &= close_enough(v1[i], v2[i]);
    }

    report("CausalMultiHeadedAttention deterministic (same input -> same output)", ok);
}

// ═══════════════════════════════════════════════════════════════════════════
//  8. LayerNorm (model/norm.h) — layer normalization
//
//  For each row (last dimension):
//    1. Compute mean and variance across n_embd
//    2. Normalize: x_hat = (x - mean) / sqrt(var + eps)
//    3. Scale and shift: output = gamma * x_hat + beta
//
//  eps = 1e-5 prevents division by zero.
//  gamma (weights) and beta (biases) are learnable per-feature parameters.
//
//  PyTorch equivalent: nn.LayerNorm(n_embd)
// ═══════════════════════════════════════════════════════════════════════════

static void test_layernorm_output_shape() {
    //  input [2, 3, 4] → output [2, 3, 4]  (shape preserved)
    LayerNorm ln({2, 3, 4});
    Tensor input({2, 3, 4}, 1.0f);
    Tensor output;
    ln.forward(input, output);

    bool ok = (output.ndim() == 3)
           && (output.shape(0) == 2)
           && (output.shape(1) == 3)
           && (output.shape(2) == 4);

    report("LayerNorm output shape [2,3,4]", ok);
}

static void test_layernorm_uniform_input() {
    //  If all elements in a row are the same, the normalized values are 0
    //  (because x - mean = 0 for every element).
    //  Then output = gamma * 0 + beta = beta for each element.
    //
    //  With random gamma/beta (seed 42), we just verify all values in a row
    //  are identical (since every input position gets the same normalization).
    LayerNorm ln({1, 1, 4});
    Tensor input({1, 1, 4}, 5.0f);
    Tensor output;
    ln.forward(input, output);

    auto vals = to_host(output);

    // Reproduce the expected biases from seed-42 init
    // (since (x - mean) = 0 for uniform input, output = gamma*0 + beta = beta)
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> expected_biases(4);
    for (int i = 0; i < 4; i++) {
        dist(rng);                     // skip gamma[i]
        expected_biases[i] = dist(rng); // beta[i]
    }

    report("LayerNorm uniform input -> output equals beta",
           check_tensor(output, expected_biases, 1e-3f));
}

static void test_layernorm_known_values() {
    //  Test with a hand-computed example.
    //  input = [[[2, 4, 6]]]  shape [1, 1, 3]
    //
    //  mean = (2+4+6)/3 = 4
    //  var  = ((2-4)^2 + (4-4)^2 + (6-4)^2) / 3 = (4+0+4)/3 = 8/3
    //  inv_std = 1 / sqrt(8/3 + 1e-5)
    //
    //  x_hat[0] = (2-4) * inv_std = -2 * inv_std
    //  x_hat[1] = (4-4) * inv_std = 0
    //  x_hat[2] = (6-4) * inv_std = 2 * inv_std
    //
    //  output = gamma * x_hat + beta
    //
    //  With seed-42 random gamma/beta, we compute expected on the host.
    LayerNorm ln({1, 1, 3});
    Tensor input({1, 1, 3});
    upload(input, {2.0f, 4.0f, 6.0f});

    Tensor output;
    ln.forward(input, output);

    // Reproduce gamma and beta from seed 42
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float gamma[3], beta[3];
    for (int i = 0; i < 3; i++) {
        gamma[i] = dist(rng);
        beta[i]  = dist(rng);
    }

    float mean = 4.0f;
    float var  = 8.0f / 3.0f;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    float in[3] = {2.0f, 4.0f, 6.0f};
    std::vector<float> expected(3);
    for (int i = 0; i < 3; i++) {
        expected[i] = gamma[i] * (in[i] - mean) * inv_std + beta[i];
    }

    report("LayerNorm known values [2,4,6]",
           check_tensor(output, expected, 1e-3f));
}

static void test_layernorm_rows_independent() {
    //  Each (batch, time) position is normalized independently.
    //  Two rows with different distributions should produce different outputs.
    LayerNorm ln({1, 2, 3});
    Tensor input({1, 2, 3});
    upload(input, {1, 2, 3, 10, 20, 30});

    Tensor output;
    ln.forward(input, output);

    auto vals = to_host(output);

    // The two rows have the same relative distribution (1:2:3 vs 10:20:30)
    // so after normalization x_hat values should be identical.
    // But since gamma/beta are the same for both rows, outputs should match.
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        ok &= close_enough(vals[i], vals[3 + i], 1e-3f);
    }

    report("LayerNorm proportional rows -> same output", ok);
}

static void test_layernorm_rejects_non_3d() {
    //  Current implementation requires exactly 3D input [B, T, C].
    LayerNorm ln({4});
    Tensor input({4}, 1.0f);
    Tensor output;

    bool threw = false;
    try {
        ln.forward(input, output);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    report("LayerNorm rejects non-3D input", threw);
}

// ═══════════════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    printf("========================================\n");
    printf("  Layer Tests\n");
    printf("========================================\n\n");

    printf("--- Tensor ---\n");
    test_tensor_construction();
    test_tensor_negative_indexing();
    test_tensor_resize();
    test_tensor_reshape();
    test_tensor_split_2way();
    test_tensor_split_3way();

    printf("\n--- ReLU ---\n");
    test_relu_basic();
    test_relu_all_negative();
    test_relu_all_positive();
    test_relu_preserves_shape();

    printf("\n--- GELU ---\n");
    test_gelu_basic();
    test_gelu_zero();
    test_gelu_large_positive();
    test_gelu_large_negative();
    test_gelu_preserves_shape();

    printf("\n--- Softmax ---\n");
    test_softmax_basic();
    test_softmax_rows_independent();
    test_softmax_sums_to_one();
    test_softmax_uniform_input();
    test_softmax_3d_tensor();

    printf("\n--- LinearLayer ---\n");
    test_linear_identity();
    test_linear_known_weights();
    test_linear_batched();
    test_linear_3d_input();
    test_linear_rejects_wrong_features();

    printf("\n--- Embedding ---\n");
    test_embedding_output_shape();
    test_embedding_same_token_same_vector();
    test_embedding_known_values();
    test_embedding_out_of_range();

    printf("\n--- Attention ---\n");
    test_attention_seq_len_1();
    test_attention_causal_mask();
    test_attention_output_shape();
    test_attention_first_row_copies_first_value();

    printf("\n--- CausalMultiHeadedAttention ---\n");
    test_causal_attention_output_shape();
    test_causal_attention_deterministic();

    printf("\n--- LayerNorm ---\n");
    test_layernorm_output_shape();
    test_layernorm_uniform_input();
    test_layernorm_known_values();
    test_layernorm_rows_independent();
    test_layernorm_rejects_non_3d();

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", g_passed, g_failed);
    printf("========================================\n");

    if (g_failed > 0) {
        printf("\nSome tests FAILED — see [FAIL] lines above.\n");
    }

    return (g_failed > 0) ? 1 : 0;
}
