#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path

LOG_PATH = Path("/root/repro-gpt2-c/.cursor/debug-b412b5.log")
SESSION_ID = "b412b5"
RUN_ID = f"probe-{int(time.time())}"
ROOT = Path("/root/repro-gpt2-c")


def _append_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": SESSION_ID,
        "runId": RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _run_cmd(cmd: str, hypothesis_id: str, label: str) -> int:
    # region agent log
    _append_log(
        hypothesis_id,
        "tools/debug_nvcc_probe.py:_run_cmd",
        "build_start",
        {"label": label, "cmd": cmd},
    )
    # endregion
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        shell=True,
        text=True,
        capture_output=True,
    )
    # region agent log
    _append_log(
        hypothesis_id,
        "tools/debug_nvcc_probe.py:_run_cmd",
        "build_end",
        {
            "label": label,
            "returncode": p.returncode,
            "stdout_tail": p.stdout[-4000:],
            "stderr_tail": p.stderr[-4000:],
        },
    )
    # endregion
    return p.returncode


def main() -> int:
    os.makedirs(LOG_PATH.parent, exist_ok=True)

    base = "nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -o train_gpt2 train_gpt2.cu model/impl/linear.cu model/impl/act.cu model/impl/embedding.cu"

    # region agent log
    _append_log(
        "H0",
        "tools/debug_nvcc_probe.py:main",
        "probe_started",
        {"runId": RUN_ID},
    )
    # endregion

    # H1: O2 optimization triggers nvcc segfault path.
    for i in range(1, 7):
        rc = _run_cmd(base, "H1", f"o2_loop_{i}")
        if rc != 0:
            break

    # H2: crash is tied specifically to train_gpt2.cu frontend path (C++20/span).
    _run_cmd(
        "nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -c train_gpt2.cu -o /tmp/train_gpt2.o",
        "H2",
        "single_tu_train",
    )
    _run_cmd(
        "nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -c model/impl/linear.cu -o /tmp/linear.o",
        "H2",
        "single_tu_linear",
    )

    # H3: O0 acts as reliable workaround, indicating toolchain optimization bug.
    for i in range(1, 4):
        _run_cmd(
            "nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O0 -o train_gpt2 train_gpt2.cu model/impl/linear.cu model/impl/act.cu model/impl/embedding.cu",
            "H3",
            f"o0_loop_{i}",
        )

    # H4: host compiler choice influences crash frequency.
    _run_cmd(
        "nvcc -ccbin /usr/bin/g++-9 -std=c++20 -O2 -o train_gpt2 train_gpt2.cu model/impl/linear.cu model/impl/act.cu model/impl/embedding.cu",
        "H4",
        "gpp9_o2_once",
    )

    # H5: C++20 frontend path is unstable; C++17 path should be stable.
    _run_cmd(
        "nvcc -ccbin /usr/bin/g++-10 -std=c++17 -O2 -c model/impl/linear.cu -o /tmp/linear_cpp17.o",
        "H5",
        "single_tu_linear_cpp17",
    )
    _run_cmd(
        "nvcc -ccbin /usr/bin/g++-10 -std=c++17 -O2 -c model/impl/embedding.cu -o /tmp/embedding_cpp17.o",
        "H5",
        "single_tu_embedding_cpp17",
    )

    # H6: nvcc toolchain itself is unhealthy even on minimal CUDA source.
    _run_cmd(
        "python3 - <<'PY'\nfrom pathlib import Path\np=Path('/tmp/min_nvcc.cu')\np.write_text('#include <cuda_runtime.h>\\n__global__ void k(){}\\nint main(){k<<<1,1>>>(); return 0;}\\n', encoding='utf-8')\nprint(p)\nPY",
        "H6",
        "write_minimal_cuda_file",
    )
    _run_cmd(
        "nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 /tmp/min_nvcc.cu -o /tmp/min_nvcc_o2",
        "H6",
        "minimal_cuda_o2_cpp20",
    )
    _run_cmd(
        "nvcc -ccbin /usr/bin/g++-10 -std=c++17 -O2 /tmp/min_nvcc.cu -o /tmp/min_nvcc_o2_cpp17",
        "H6",
        "minimal_cuda_o2_cpp17",
    )

    # H7: Split compilation avoids nvcc multi-TU instability.
    _run_cmd(
        "mkdir -p /tmp/repro_objs",
        "H7",
        "prep_obj_dir",
    )
    for i in range(1, 4):
        _run_cmd(
            "nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -c train_gpt2.cu -o /tmp/repro_objs/train_gpt2.o",
            "H7",
            f"split_{i}_compile_train",
        )
        _run_cmd(
            "nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -c model/impl/linear.cu -o /tmp/repro_objs/linear.o",
            "H7",
            f"split_{i}_compile_linear",
        )
        _run_cmd(
            "nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -c model/impl/act.cu -o /tmp/repro_objs/act.o",
            "H7",
            f"split_{i}_compile_act",
        )
        _run_cmd(
            "nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -c model/impl/embedding.cu -o /tmp/repro_objs/embedding.o",
            "H7",
            f"split_{i}_compile_embedding",
        )
        _run_cmd(
            "nvcc -ccbin /usr/bin/g++-10 -std=c++20 -O2 -o /tmp/repro_objs/train_gpt2_split /tmp/repro_objs/train_gpt2.o /tmp/repro_objs/linear.o /tmp/repro_objs/act.o /tmp/repro_objs/embedding.o",
            "H7",
            f"split_{i}_link",
        )

    # region agent log
    _append_log(
        "H0",
        "tools/debug_nvcc_probe.py:main",
        "probe_finished",
        {"runId": RUN_ID},
    )
    # endregion
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

