from transformers import GPT2LMHeadModel
import numpy as np
import struct


mapping = {
    "transformer.wte.weight": "embedding1.weights",
    "transformer.wpe.weight": "embedding2.weights",
    "transformer.h.0.ln_1.weight": "blocks.0.ln1.weight",
    "transformer.h.0.ln_1.bias": "blocks.0.ln1.bias",
    "transformer.h.0.attn.c_attn.weight": "blocks.0.attn.c_attn.weight",
    "transformer.h.0.attn.c_attn.bias": "blocks.0.attn.c_attn.bias",
    "transformer.h.0.attn.c_proj.weight": "blocks.0.attn.c_proj.weight",
    "transformer.h.0.attn.c_proj.bias": "blocks.0.attn.c_proj.bias",
    "transformer.h.0.ln_2.weight": "blocks.0.ln2.weight",
    "transformer.h.0.ln_2.bias": "blocks.0.ln2.bias",
    "transformer.h.0.mlp.c_fc.weight": "blocks.0.mlp.c_fc.weight",
    "transformer.h.0.mlp.c_fc.bias": "blocks.0.mlp.c_fc.bias",
    "transformer.h.0.mlp.c_proj.weight": "blocks.0.mlp.c_proj.weight",
    "transformer.h.0.mlp.c_proj.bias": "blocks.0.mlp.c_proj.bias",
    "transformer.h.1.ln_1.weight": "blocks.1.ln1.weight",
    "transformer.h.1.ln_1.bias": "blocks.1.ln1.bias",
    "transformer.h.1.attn.c_attn.weight": "blocks.1.attn.c_attn.weight",
    "transformer.h.1.attn.c_attn.bias": "blocks.1.attn.c_attn.bias",
    "transformer.h.1.attn.c_proj.weight": "blocks.1.attn.c_proj.weight",
    "transformer.h.1.attn.c_proj.bias": "blocks.1.attn.c_proj.bias",
    "transformer.h.1.ln_2.weight": "blocks.1.ln2.weight",
    "transformer.h.1.ln_2.bias": "blocks.1.ln2.bias",
    "transformer.h.1.mlp.c_fc.weight": "blocks.1.mlp.c_fc.weight",
    "transformer.h.1.mlp.c_fc.bias": "blocks.1.mlp.c_fc.bias",
    "transformer.h.1.mlp.c_proj.weight": "blocks.1.mlp.c_proj.weight",
    "transformer.h.1.mlp.c_proj.bias": "blocks.1.mlp.c_proj.bias",
    "transformer.h.2.ln_1.weight": "blocks.2.ln1.weight",
    "transformer.h.2.ln_1.bias": "blocks.2.ln1.bias",
    "transformer.h.2.attn.c_attn.weight": "blocks.2.attn.c_attn.weight",
    "transformer.h.2.attn.c_attn.bias": "blocks.2.attn.c_attn.bias",
    "transformer.h.2.attn.c_proj.weight": "blocks.2.attn.c_proj.weight",
    "transformer.h.2.attn.c_proj.bias": "blocks.2.attn.c_proj.bias",
    "transformer.h.2.ln_2.weight": "blocks.2.ln2.weight",
    "transformer.h.2.ln_2.bias": "blocks.2.ln2.bias",
    "transformer.h.2.mlp.c_fc.weight": "blocks.2.mlp.c_fc.weight",
    "transformer.h.2.mlp.c_fc.bias": "blocks.2.mlp.c_fc.bias",
    "transformer.h.2.mlp.c_proj.weight": "blocks.2.mlp.c_proj.weight",
    "transformer.h.2.mlp.c_proj.bias": "blocks.2.mlp.c_proj.bias",
    "transformer.h.3.ln_1.weight": "blocks.3.ln1.weight",
    "transformer.h.3.ln_1.bias": "blocks.3.ln1.bias",
    "transformer.h.3.attn.c_attn.weight": "blocks.3.attn.c_attn.weight",
    "transformer.h.3.attn.c_attn.bias": "blocks.3.attn.c_attn.bias",
    "transformer.h.3.attn.c_proj.weight": "blocks.3.attn.c_proj.weight",
    "transformer.h.3.attn.c_proj.bias": "blocks.3.attn.c_proj.bias",
    "transformer.h.3.ln_2.weight": "blocks.3.ln2.weight",
    "transformer.h.3.ln_2.bias": "blocks.3.ln2.bias",
    "transformer.h.3.mlp.c_fc.weight": "blocks.3.mlp.c_fc.weight",
    "transformer.h.3.mlp.c_fc.bias": "blocks.3.mlp.c_fc.bias",
    "transformer.h.3.mlp.c_proj.weight": "blocks.3.mlp.c_proj.weight",
    "transformer.h.3.mlp.c_proj.bias": "blocks.3.mlp.c_proj.bias",
    "transformer.h.4.ln_1.weight": "blocks.4.ln1.weight",
    "transformer.h.4.ln_1.bias": "blocks.4.ln1.bias",
    "transformer.h.4.attn.c_attn.weight": "blocks.4.attn.c_attn.weight",
    "transformer.h.4.attn.c_attn.bias": "blocks.4.attn.c_attn.bias",
    "transformer.h.4.attn.c_proj.weight": "blocks.4.attn.c_proj.weight",
    "transformer.h.4.attn.c_proj.bias": "blocks.4.attn.c_proj.bias",
    "transformer.h.4.ln_2.weight": "blocks.4.ln2.weight",
    "transformer.h.4.ln_2.bias": "blocks.4.ln2.bias",
    "transformer.h.4.mlp.c_fc.weight": "blocks.4.mlp.c_fc.weight",
    "transformer.h.4.mlp.c_fc.bias": "blocks.4.mlp.c_fc.bias",
    "transformer.h.4.mlp.c_proj.weight": "blocks.4.mlp.c_proj.weight",
    "transformer.h.4.mlp.c_proj.bias": "blocks.4.mlp.c_proj.bias",
    "transformer.h.5.ln_1.weight": "blocks.5.ln1.weight",
    "transformer.h.5.ln_1.bias": "blocks.5.ln1.bias",
    "transformer.h.5.attn.c_attn.weight": "blocks.5.attn.c_attn.weight",    
    "transformer.h.5.attn.c_attn.bias": "blocks.5.attn.c_attn.bias",
    "transformer.h.5.attn.c_proj.weight": "blocks.5.attn.c_proj.weight",
    "transformer.h.5.attn.c_proj.bias": "blocks.5.attn.c_proj.bias",
    "transformer.h.5.ln_2.weight": "blocks.5.ln2.weight",
    "transformer.h.5.ln_2.bias": "blocks.5.ln2.bias",
    "transformer.h.5.mlp.c_fc.weight": "blocks.5.mlp.c_fc.weight",
    "transformer.h.5.mlp.c_fc.bias": "blocks.5.mlp.c_fc.bias",
    "transformer.h.5.mlp.c_proj.weight": "blocks.5.mlp.c_proj.weight",
    "transformer.h.5.mlp.c_proj.bias": "blocks.5.mlp.c_proj.bias",
    "transformer.h.6.ln_1.weight": "blocks.6.ln1.weight",
    "transformer.h.6.ln_1.bias": "blocks.6.ln1.bias",
    "transformer.h.6.attn.c_attn.weight": "blocks.6.attn.c_attn.weight",
    "transformer.h.6.attn.c_attn.bias": "blocks.6.attn.c_attn.bias",
    "transformer.h.6.attn.c_proj.weight": "blocks.6.attn.c_proj.weight",
    "transformer.h.6.attn.c_proj.bias": "blocks.6.attn.c_proj.bias",
    "transformer.h.6.ln_2.weight": "blocks.6.ln2.weight",
    "transformer.h.6.ln_2.bias": "blocks.6.ln2.bias",
    "transformer.h.6.mlp.c_fc.weight": "blocks.6.mlp.c_fc.weight",
    "transformer.h.6.mlp.c_fc.bias": "blocks.6.mlp.c_fc.bias",
    "transformer.h.6.mlp.c_proj.weight": "blocks.6.mlp.c_proj.weight",
    "transformer.h.6.mlp.c_proj.bias": "blocks.6.mlp.c_proj.bias",
    "transformer.h.7.ln_1.weight": "blocks.7.ln1.weight",
    "transformer.h.7.ln_1.bias": "blocks.7.ln1.bias",
    "transformer.h.7.attn.c_attn.weight": "blocks.7.attn.c_attn.weight",
    "transformer.h.7.attn.c_attn.bias": "blocks.7.attn.c_attn.bias",
    "transformer.h.7.attn.c_proj.weight": "blocks.7.attn.c_proj.weight",
    "transformer.h.7.attn.c_proj.bias": "blocks.7.attn.c_proj.bias",
    "transformer.h.7.ln_2.weight": "blocks.7.ln2.weight",
    "transformer.h.7.ln_2.bias": "blocks.7.ln2.bias",
    "transformer.h.7.mlp.c_fc.weight": "blocks.7.mlp.c_fc.weight",
    "transformer.h.7.mlp.c_fc.bias": "blocks.7.mlp.c_fc.bias",
    "transformer.h.7.mlp.c_proj.weight": "blocks.7.mlp.c_proj.weight",
    "transformer.h.7.mlp.c_proj.bias": "blocks.7.mlp.c_proj.bias",
    "transformer.h.8.ln_1.weight": "blocks.8.ln1.weight",
    "transformer.h.8.ln_1.bias": "blocks.8.ln1.bias",
    "transformer.h.8.attn.c_attn.weight": "blocks.8.attn.c_attn.weight",
    "transformer.h.8.attn.c_attn.bias": "blocks.8.attn.c_attn.bias",
    "transformer.h.8.attn.c_proj.weight": "blocks.8.attn.c_proj.weight",
    "transformer.h.8.attn.c_proj.bias": "blocks.8.attn.c_proj.bias",
    "transformer.h.8.ln_2.weight": "blocks.8.ln2.weight",
    "transformer.h.8.ln_2.bias": "blocks.8.ln2.bias",
    "transformer.h.8.mlp.c_fc.weight": "blocks.8.mlp.c_fc.weight",
    "transformer.h.8.mlp.c_fc.bias": "blocks.8.mlp.c_fc.bias",
    "transformer.h.8.mlp.c_proj.weight": "blocks.8.mlp.c_proj.weight",
    "transformer.h.8.mlp.c_proj.bias": "blocks.8.mlp.c_proj.bias",
    "transformer.h.9.ln_1.weight": "blocks.9.ln1.weight",
    "transformer.h.9.ln_1.bias": "blocks.9.ln1.bias",
    "transformer.h.9.attn.c_attn.weight": "blocks.9.attn.c_attn.weight",
    "transformer.h.9.attn.c_attn.bias": "blocks.9.attn.c_attn.bias",
    "transformer.h.9.attn.c_proj.weight": "blocks.9.attn.c_proj.weight",
    "transformer.h.9.attn.c_proj.bias": "blocks.9.attn.c_proj.bias",
    "transformer.h.9.ln_2.weight": "blocks.9.ln2.weight",
    "transformer.h.9.ln_2.bias": "blocks.9.ln2.bias",
    "transformer.h.9.mlp.c_fc.weight": "blocks.9.mlp.c_fc.weight",
    "transformer.h.9.mlp.c_fc.bias": "blocks.9.mlp.c_fc.bias",
    "transformer.h.9.mlp.c_proj.weight": "blocks.9.mlp.c_proj.weight",
    "transformer.h.9.mlp.c_proj.bias": "blocks.9.mlp.c_proj.bias",
    "transformer.h.10.ln_1.weight": "blocks.10.ln1.weight",
    "transformer.h.10.ln_1.bias": "blocks.10.ln1.bias",
    "transformer.h.10.attn.c_attn.weight": "blocks.10.attn.c_attn.weight",
    "transformer.h.10.attn.c_attn.bias": "blocks.10.attn.c_attn.bias",
    "transformer.h.10.attn.c_proj.weight": "blocks.10.attn.c_proj.weight",
    "transformer.h.10.attn.c_proj.bias": "blocks.10.attn.c_proj.bias",
    "transformer.h.10.ln_2.weight": "blocks.10.ln2.weight",
    "transformer.h.10.ln_2.bias": "blocks.10.ln2.bias",
    "transformer.h.10.mlp.c_fc.weight": "blocks.10.mlp.c_fc.weight",
    "transformer.h.10.mlp.c_fc.bias": "blocks.10.mlp.c_fc.bias",
    "transformer.h.10.mlp.c_proj.weight": "blocks.10.mlp.c_proj.weight",
    "transformer.h.10.mlp.c_proj.bias": "blocks.10.mlp.c_proj.bias",
    "transformer.h.11.ln_1.weight": "blocks.11.ln1.weight",
    "transformer.h.11.ln_1.bias": "blocks.11.ln1.bias",
    "transformer.h.11.attn.c_attn.weight": "blocks.11.attn.c_attn.weight",
    "transformer.h.11.attn.c_attn.bias": "blocks.11.attn.c_attn.bias",
    "transformer.h.11.attn.c_proj.weight": "blocks.11.attn.c_proj.weight",
    "transformer.h.11.attn.c_proj.bias": "blocks.11.attn.c_proj.bias",
    "transformer.h.11.ln_2.weight": "blocks.11.ln2.weight",
    "transformer.h.11.ln_2.bias": "blocks.11.ln2.bias",
    "transformer.h.11.mlp.c_fc.weight": "blocks.11.mlp.c_fc.weight",
    "transformer.h.11.mlp.c_fc.bias": "blocks.11.mlp.c_fc.bias",
    "transformer.h.11.mlp.c_proj.weight": "blocks.11.mlp.c_proj.weight",
    "transformer.h.11.mlp.c_proj.bias": "blocks.11.mlp.c_proj.bias",
    "transformer.ln_f.weight": "ln_out.weight",
    "transformer.ln_f.bias": "ln_out.bias",
    "lm_head.weight": "linear_out.weights"
}

if __name__ == "__main__":
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    weights = model_hf.state_dict()

    records = {}
    for hf_key, t in weights.items():
        if hf_key not in mapping:
            raise ValueError(f"Key {hf_key} not in mapping")

        out_key = mapping[hf_key]

        a = t.detach().cpu().numpy()

        # Match CUDA LinearLayer weight layout: [out_features, in_features].
        # HF GPT-2 uses Conv1D for blocks; Conv1D stores weight as [in_features, out_features],
        # so those need a transpose. lm_head is nn.Linear — weight is already [out, in] like
        # PyTorch nn.Linear and must NOT be transposed.
        if hf_key.endswith(".weight") and a.ndim == 2 and hf_key not in (
            "transformer.wte.weight",
            "transformer.wpe.weight",
            "lm_head.weight",
        ) and (".ln_" not in hf_key) and (not hf_key.startswith("transformer.ln_f.")):
            a = a.T

        a = np.asarray(a, dtype=np.float32, order="C")
        records[out_key] = (tuple(int(x) for x in a.shape), a.tobytes(order="C"))

    # Binary file layout (little-endian):
    #   count    = uint32 (#tensors)
    # Then repeated 'count' times:
    #   key_len     uint32
    #   key_bytes   key_len bytes (utf-8, no terminator)
    #   dtype       uint32 (1 = float32)
    #   ndim        uint32
    #   shape       int32[ndim]
    #   data_nbytes uint64
    #   data        data_nbytes bytes (raw float32, row-major)
    dtype_code = 1  # float32
    keys = sorted(records.keys())
    with open("weights.bin", "wb") as f:
        f.write(struct.pack("<I", len(keys)))
        for k in keys:
            shape, data = records[k]
            kb = k.encode("utf-8")
            f.write(struct.pack("<I", len(kb)))
            f.write(kb)
            f.write(struct.pack("<I", dtype_code))
            f.write(struct.pack("<I", len(shape)))
            f.write(struct.pack("<" + "i" * len(shape), *shape))
            f.write(struct.pack("<Q", len(data)))
            f.write(data)