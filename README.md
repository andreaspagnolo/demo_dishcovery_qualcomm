# Dishcovery on Qualcomm Dragonwing IQ-9075 EVK

## Reproduction baseline and optimization targets

This repository is the fixed Qualcomm baseline for the Dishcovery Task 1 and Task 2 pipelines. It has four purposes:

1. provide a complete, copy-and-run procedure for reproducing the current 350-image Qualcomm results on a Dragonwing IQ-9075 EVK;
2. pin the model artifacts, library versions, native bridge revision, runtime environment, and benchmark commands used by the validated baseline;
3. document the current latency bottlenecks in SigLIP2 and Qwen3-VL-Reranker-2B, including how the deployed artifacts were produced and why those implementations were selected;
4. preserve the controlled Orin-versus-EVK Task 1 accuracy investigation, which shows that the remaining cross-platform F1 gap is concentrated in the Qwen3-VL-4B-Instruct deployment rather than in SigLIP2 retrieval or caller-side image preparation.

The goal is to let Qualcomm engineers reproduce the baseline, then investigate faster and more accurate replacement implementations. The target is to approach the accuracy/F1 and latency obtained on NVIDIA Jetson AGX Orin while preserving the benchmark contract defined below.

## Contents

- [Part 1 — Reproduce the current results](#part-1--reproduce-the-current-results)
- [Part 2A — Known latency problems and model-build rationale](#part-2a--known-latency-problems-and-model-build-rationale)
- [Part 2B — Qwen3-VL-4B-Instruct Task 1 accuracy regression](#part-2b--qwen3-vl-4b-instruct-task-1-accuracy-regression)

---

# Part 1 — Reproduce the current results

## 1. Repository and asset layout

Git contains all source code, small inputs, cached text embeddings, configuration files, logs, and reference results. The private asset package supplies the 350-image data, the fixed SigLIP2 and Qwen3-VL-Reranker-2B binaries, the Qualcomm ONNX Runtime/QNN wheels, and a pinned GenieX-compatible Qwen3-VL-4B-Instruct W4A16 bundle. Keeping these external artifacts in one versioned private package avoids depending on mutable online model names or wheel locations during reproduction.

Download the private asset package from [Google Drive](https://drive.google.com/drive/folders/1gGnXaYtdx4e8cTYwCkAXPakYQqqiBwc5?usp=drive_link), then extract it at the repository root so that it creates exactly this layout:

```text
external_assets/
├── images/
│   ├── task1_350/                         # 350 img_*.jpg files
│   ├── task2_350/                         # Food-500 class/image hierarchy
│   └── demo/                              # browser gallery, up to 80 images
├── wheels/
│   ├── onnxruntime-1.24.4-*.whl          # Qualcomm aarch64 wheel
│   └── onnxruntime_qnn-2.1.0-*.whl       # ORT-QNN / QAIRT 2.45.40 wheel
└── models/
    ├── siglip2_qcs9075_out/fp16_powfix_split_qairt245/
    │   ├── stage1/model.onnx
    │   ├── stage1/model.bin
    │   ├── stage2/model.onnx
    │   └── stage2/model.bin
    ├── Qwen3-VL-Reranker-2B-GGUF/
    │   ├── Qwen3-VL-Reranker-2B-Q8_0-F16Emb-F16Cls-HTP.gguf
    │   └── mmproj-Qwen3-VL-Reranker-2B-F16.gguf
    └── Qwen3-VL-4B-Instruct-GenieX-QAIRT-W4A16.zip
```

Verify all versioned inputs and Drive-delivered model artifacts:

```bash
python3 scripts/verify_external_assets.py
```

Every entry must be reported as `OK`. The expected SHA-256 values are committed in:

```text
config/checksums/model_and_input_sha256.txt
```

The pinned Qwen3-VL-4B-Instruct bundle and Qualcomm wheel files should also be covered by the private package checksum manifest distributed with the assets. Do not silently replace them with newer artifacts when reproducing the baseline.

## 2. Repository map

```text
benchmark_inputs/       Versioned lists, labels, captions, mappings, and frozen text embeddings
config/                 Checksums, platform contract, model metadata, and benchmark settings
external_assets/        Ignored private-asset target: images, wheels, and pinned model bundles
model_build/            SigLIP2 graph/compile utilities and reranker MTMD/Q8 build code
pipeline/               Task 1 and Task 2 evaluation code; calorie support used by the demo
demo_web/               Browser demo server, backend, and static UI
reference_results/      Archived summaries, JSON, CSV, checkpoints, and logs
reports/                Optimization experiments, manifests, raw outputs, and comparisons
scripts/                Fixed verification, benchmark, and demo entry points
```

## 3. Recorded platform and software stack

Reference platform:

```text
Device:                   Qualcomm Dragonwing IQ-9075 EVK
Architecture:             aarch64
GenieX:                   0.3.13
QAIRT used by GenieX:     2.45.0.260326
ONNX Runtime:             1.24.4
onnxruntime-qnn:          2.1.0 / QAIRT 2.45.40
llama.cpp revision:       4f31eedb0ccf546b7e8d6bb243b170f12522f54d
```

Do not mix QAIRT 2.47 ONNX Runtime/QNN libraries with the GenieX 2.45 runtime used by this baseline.

### 3.1 Install the system prerequisites

Run once on the EVK:

```bash
sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  git cmake ninja-build build-essential pkg-config \
  ocl-icd-opencl-dev
```

Run all remaining commands from the repository root. After opening a shell in the cloned repository, verify it before continuing:

```bash
test -f requirements.txt && test -d scripts && echo "Repository root: OK"
```

### 3.2 Create the Python environment

`requirements.txt` may temporarily install the public ONNX Runtime package as a transitive dependency. The commands below intentionally replace it with the validated Qualcomm pair afterward.

```bash
python3 -m venv .venv_ort_qnn_245
export EVK_PYTHON="$PWD/.venv_ort_qnn_245/bin/python"

"$EVK_PYTHON" -m pip install --upgrade pip
"$EVK_PYTHON" -m pip install -r requirements.txt
```

### 3.3 Install the Qualcomm ONNX Runtime and ORT-QNN wheels

The private asset package must contain exactly one matching wheel for each component under `external_assets/wheels/`.

```bash
mapfile -t ORT_WHEELS < <(
  find external_assets/wheels -maxdepth 1 -type f \
    -name 'onnxruntime-1.24.4-*.whl' | sort
)

mapfile -t ORT_QNN_WHEELS < <(
  find external_assets/wheels -maxdepth 1 -type f \
    \( -name 'onnxruntime_qnn-2.1.0-*.whl' \
       -o -name 'onnxruntime-qnn-2.1.0-*.whl' \) | sort
)

[ "${#ORT_WHEELS[@]}" -eq 1 ] || {
  echo "ERROR: expected exactly one ONNX Runtime 1.24.4 wheel" >&2
  printf '%s\n' "${ORT_WHEELS[@]}"
  exit 1
}

[ "${#ORT_QNN_WHEELS[@]}" -eq 1 ] || {
  echo "ERROR: expected exactly one ORT-QNN 2.1.0 wheel" >&2
  printf '%s\n' "${ORT_QNN_WHEELS[@]}"
  exit 1
}

# Remove the public/transitive ORT package before installing the validated pair.
"$EVK_PYTHON" -m pip uninstall -y \
  onnxruntime onnxruntime-gpu onnxruntime-qnn

"$EVK_PYTHON" -m pip install --no-cache-dir \
  "${ORT_WHEELS[0]}" \
  "${ORT_QNN_WHEELS[0]}"
```

Verify the Python packages, QNN plugin, and HTP libraries before continuing:

```bash
"$EVK_PYTHON" - <<'PY'
from importlib import metadata
from pathlib import Path

import onnxruntime as ort
import onnxruntime_qnn as qnn

try:
    qnn_version = metadata.version("onnxruntime-qnn")
except metadata.PackageNotFoundError:
    qnn_version = metadata.version("onnxruntime_qnn")

assert ort.__version__ == "1.24.4", ort.__version__
assert qnn_version == "2.1.0", qnn_version

provider = qnn.get_ep_name()
devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
if not devices:
    ort.register_execution_provider_library(provider, qnn.get_library_path())
    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]

assert devices, f"{provider} was not discovered"
assert Path(qnn.get_library_path()).is_file(), qnn.get_library_path()
assert Path(qnn.get_qnn_htp_path()).is_file(), qnn.get_qnn_htp_path()

print("ONNX Runtime:", ort.__version__)
print("ORT-QNN:", qnn_version)
print("QNN provider:", provider)
print("QNN devices:", devices)
print("QNN plugin:", qnn.get_library_path())
print("QNN HTP backend:", qnn.get_qnn_htp_path())
PY
```

A warning about `/sys/class/drm/card0/device/vendor` is not fatal on this EVK. The required condition is that the script reaches the final version, provider, device, and library lines without an assertion failure.

### 3.4 Export the baseline paths

```bash
export TASK1_PYTHON="$PWD/.venv_ort_qnn_245/bin/python"
export TASK2_PYTHON="$PWD/.venv_ort_qnn_245/bin/python"
export GENIEX_QNN_BACKEND="$HOME/.local/share/geniex/qairt/htp-files/libQnnHtp.so"
export DISHCOVERY_MODELS_DIR="$PWD/external_assets/models"

for path in \
  "$TASK1_PYTHON" \
  "$TASK2_PYTHON" \
  "$GENIEX_QNN_BACKEND" \
  "$DISHCOVERY_MODELS_DIR"; do
  test -e "$path" || { echo "ERROR: missing $path" >&2; exit 1; }
done
```

Confirm the pinned GenieX stack:

```bash
geniex --version
```

The output must identify GenieX `0.3.13` and QAIRT `2.45.0.260326`.

The fixed platform policy is recorded in:

```text
config/platform/evk_qairt_stack.json
```

The Task 2 runner validates this contract strictly.

## 4. Import Qwen3-VL-4B-Instruct for Task 1

Import the bundle supplied in `external_assets/models/`:

```bash
export QWEN_GENIEX_BUNDLE="$PWD/external_assets/models/Qwen3-VL-4B-Instruct-GenieX-QAIRT-W4A16.zip"
export GENIEX_TASK1_MODEL="local/qwen3vl-4b-qairt-w4a16"

test -f "$QWEN_GENIEX_BUNDLE" || {
  echo "ERROR: missing pinned Qwen3-VL-4B-Instruct GenieX bundle" >&2
  exit 1
}

if ! geniex list | grep -Fq "$GENIEX_TASK1_MODEL"; then
  geniex pull "$GENIEX_TASK1_MODEL" \
    --model-hub localfs \
    --local-path "$QWEN_GENIEX_BUNDLE" \
    --model-type vlm
fi

geniex list | grep -F "$GENIEX_TASK1_MODEL"
```

The final command must print:

```text
local/qwen3vl-4b-qairt-w4a16
```

Do not update or replace the model bundle during a baseline reproduction. Record a new artifact name and hash when evaluating a replacement.

## 5. Build the Qwen3-VL-Reranker-2B MTMD/llama.cpp bridge

The reference reranker is not executed through ordinary causal-generation logits. It is a rank-pooling model whose two-output classifier must be preserved. The native bridge combines MTMD multimodal processing with llama.cpp rank pooling.

### 5.1 Clone the exact llama.cpp revision

Run from the repository root:

```bash
mkdir -p third_party

if [ ! -d third_party/llama.cpp/.git ]; then
  git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
fi

git -C third_party/llama.cpp fetch --all --tags
git -C third_party/llama.cpp checkout --detach \
  4f31eedb0ccf546b7e8d6bb243b170f12522f54d

export LLAMA_CPP_SOURCE="$PWD/third_party/llama.cpp"

test "$(git -C "$LLAMA_CPP_SOURCE" rev-parse HEAD)" = \
  "4f31eedb0ccf546b7e8d6bb243b170f12522f54d" \
  && echo "llama.cpp revision: OK"
```

### 5.2 Locate the ABI-compatible GenieX libraries

```bash
export GENIEX_LLAMA_LIB_DIR="$HOME/.local/share/geniex/llama_cpp"
export QAIRT_HTP_DIR="$HOME/.local/share/geniex/qairt/htp-files"

test -d "$GENIEX_LLAMA_LIB_DIR" || {
  echo "ERROR: GenieX llama.cpp library directory not found" >&2
  exit 1
}

test -d "$QAIRT_HTP_DIR" || {
  echo "ERROR: GenieX QAIRT HTP directory not found" >&2
  exit 1
}

ls "$GENIEX_LLAMA_LIB_DIR"/libllama.so* >/dev/null
ls "$GENIEX_LLAMA_LIB_DIR"/libggml-opencl.so >/dev/null
ls "$GENIEX_LLAMA_LIB_DIR"/libggml-htp-v*.so >/dev/null
```

When GenieX is installed in a non-standard directory, locate the libraries and update the two variables before building:

```bash
find "$HOME/.local/share/geniex" -type f \
  \( -name 'libllama.so*' -o -name 'libQnnHtp.so' \) \
  -printf '%h\n' 2>/dev/null | sort -u
```

### 5.3 Build and validate the bridge

```bash
rm -rf model_build/qwen3_vl_reranker/build
./model_build/qwen3_vl_reranker/build.sh

export RERANKER_BIN="$PWD/model_build/qwen3_vl_reranker/build/qwen3-vl-reranker-mtmd"

test -x "$RERANKER_BIN" || {
  echo "ERROR: Task 2 bridge was not created" >&2
  exit 1
}

if ldd "$RERANKER_BIN" | grep -q 'not found'; then
  ldd "$RERANKER_BIN" | grep 'not found'
  echo "ERROR: Task 2 bridge has missing shared libraries" >&2
  exit 1
fi
```

### 5.4 Export the Task 2 runtime library paths and check devices

These exports are required at runtime, not only while building:

```bash
export LD_LIBRARY_PATH="$GENIEX_LLAMA_LIB_DIR:$QAIRT_HTP_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$GENIEX_LLAMA_LIB_DIR;$QAIRT_HTP_DIR"
export DSP_LIBRARY_PATH="$ADSP_LIBRARY_PATH"

"$RERANKER_BIN" --list-devices
```

The device list must include all three backends:

```text
GPUOpenCL   QUALCOMM Adreno(TM) 663
HTP0        Hexagon
CPU         CPU
```

Warnings stating that `libOpenCL.so.1` has no version information are non-fatal when `GPUOpenCL` is present in the device list.

At runtime:

- MTMD executes image encoding and the multimodal projector;
- llama.cpp executes rank pooling and returns the binary relevance score;
- the text model uses the Qualcomm hybrid/HTP-capable route;
- the FP16 vision/projector path uses the exposed MTMD Adreno OpenCL route;
- `vision.UPSCALE` has one CPU fallback.

## 6. Run the exact 350-image benchmarks

Run the two tasks sequentially. **The GenieX server is required for Task 1 and must be stopped before Task 2.** Task 2 does not use the GenieX HTTP server; it directly loads the GenieX llama.cpp/QAIRT libraries through the native reranker bridge.

Before starting, export the shared baseline variables from the repository root:

```bash
export TASK1_PYTHON="$PWD/.venv_ort_qnn_245/bin/python"
export TASK2_PYTHON="$PWD/.venv_ort_qnn_245/bin/python"
export GENIEX_QNN_BACKEND="$HOME/.local/share/geniex/qairt/htp-files/libQnnHtp.so"
export DISHCOVERY_MODELS_DIR="$PWD/external_assets/models"
export GENIEX_TASK1_MODEL="local/qwen3vl-4b-qairt-w4a16"
```

### 6.1 Task 1 — start and keep `geniex serve` active

Open **Terminal A** in the repository root, then run:

```bash
test -f requirements.txt && test -d scripts || {
  echo "ERROR: Terminal A is not in the repository root" >&2
  exit 1
}

export GENIEX_TASK1_MODEL="local/qwen3vl-4b-qairt-w4a16"
geniex serve --host 127.0.0.1:18181
```

Leave Terminal A open. Then run Task 1 in **Terminal B**, from the repository root:

```bash
# Task 1: first 350 images, seed 7, legacy fixed Top-20 policy.
python3 scripts/run_350_benchmarks.py task1 \
  --qnn-backend-path "$GENIEX_QNN_BACKEND"

python3 scripts/verify_metrics.py task1 \
  run_outputs/task1_350/eval_first_350_samples.json
```

Expected verification output:

```text
f1: actual=0.660054 expected=0.660054 delta=+0.000000
precision: actual=0.757764 expected=0.757764 delta=+0.000000
recall: actual=0.584665 expected=0.584665 delta=+0.000000
```

### 6.2 Stop `geniex serve` before Task 2

Return to Terminal A and press:

```text
Ctrl+C
```

Confirm that no GenieX server remains active:

```bash
if pgrep -af 'geniex serve'; then
  echo "ERROR: stop geniex serve before running Task 2" >&2
  exit 1
fi
```

Stopping the server releases the VLM resources before the Task 2 reranker opens its own HTP/OpenCL sessions.

### 6.3 Task 2 — run with the GenieX server stopped

In Terminal B, export the native reranker runtime paths if they are not already present:

```bash
export GENIEX_LLAMA_LIB_DIR="$HOME/.local/share/geniex/llama_cpp"
export QAIRT_HTP_DIR="$HOME/.local/share/geniex/qairt/htp-files"
export LD_LIBRARY_PATH="$GENIEX_LLAMA_LIB_DIR:$QAIRT_HTP_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$GENIEX_LLAMA_LIB_DIR;$QAIRT_HTP_DIR"
export DSP_LIBRARY_PATH="$ADSP_LIBRARY_PATH"
```

Run and verify Task 2:

```bash
# Task 2: first 350 images, seed 42, class_caption evaluation,
# SigLIP Top-5, guarded Q8 reranker Top-5, gap 3.0,
# and five independent reranker calls.
python3 scripts/run_350_benchmarks.py task2 \
  --qnn-backend-path "$GENIEX_QNN_BACKEND"

python3 scripts/verify_metrics.py task2 \
  run_outputs/task2_350/eval_first_350_caption_alignment.json
```

Expected verification output:

```text
top1_caption_accuracy: actual=0.677143 expected=0.677143 delta=+0.000000
class_top1_accuracy: actual=0.877143 expected=0.877143 delta=+0.000000
```

Expected metrics:

| Task | Metric | Expected result |
| --- | --- | ---: |
| Task 1 | micro-F1 | 0.660054 |
| Task 1 | precision / recall | 0.757764 / 0.584665 |
| Task 2 | caption accuracy | 0.677143 (237/350) |
| Task 2 | class accuracy | 0.877143 (307/350) |

The fixed Task 1 contract includes:

```text
Dataset order:             first 350 images
Seed:                      7
Candidate policy:          legacy fixed Top-20
VLM output policy:         present_possible
Maximum generated tokens: 96
Image input:               512 x 512 white letterbox
SigLIP2 fusion weight:     0.25
Qwen fusion weight:        0.75
Final selector:            global row-z threshold_ratio
Selector parameters:       3.5, 0.85, maximum 7 labels
VLM skip relative gap:     0.25
```

The fixed Task 2 contract includes:

```text
Dataset order:             first 350 images
Seed:                      42
Evaluation mode:           class_caption
SigLIP candidates:         Top-5
Reranker policy:           guarded Q8 Top-5
Guard gap:                 3.0
Reranker calls:            five independent candidate calls
```

Fresh logs and complete JSON/CSV outputs are written to `run_outputs/`.

Reference evidence is stored in:

```text
reference_results/350_subset_summary.md
reference_results/task1_350/
reference_results/task2_350/
```

Latency values are reference evidence, not a strict pass/fail requirement when power mode, thermal state, or system load differs. Accuracy metrics and the fixed evaluation contract must remain unchanged.

## 7. Run the browser demo

The browser demo retains the current default policy: Task 1 preload, SigLIP-guarded Q8 MTMD Task 2 when requested, and gallery images from `external_assets/images/demo`.

```bash
python3 scripts/start_demo.py
```

Open:

```text
http://127.0.0.1:8787
```

## 8. Validation requirements for replacement implementations

For every candidate optimization, record at least:

- model and artifact hashes;
- GenieX, QAIRT, ONNX Runtime, QNN, and llama.cpp versions;
- HTP/GPU/CPU placement and every fallback;
- end-to-end and per-stage latency;
- peak and incremental RSS;
- Task 1 micro-F1, precision, recall, and candidate recall;
- Task 2 caption and class accuracy;
- guarded-row and reranked-pair counts;
- failures, malformed outputs, schema errors, and restarts.

A faster artifact is not a valid replacement until it stays within the stated accuracy tolerance under the unchanged benchmark protocol.

---

# Part 2A — Known latency problems and model-build rationale

This section explains how the two current Qualcomm-side bottlenecks were deployed, which alternatives were tested, and why the present implementations were selected. These implementations are reproducible baselines, not claimed performance optima.

## 9. SigLIP2 visual encoder

### 9.1 Deployed model contract

The selected retrieval model is the visual encoder of:

```text
timm/ViT-gopt-16-SigLIP2-384
```

Its fixed EVK input/output contract is:

```text
Input color:       RGB
Resize:            bicubic, 384 x 384
Tensor layout:     NCHW
Input dtype:       float32
Mean/std:          [0.5, 0.5, 0.5]
Output:            one L2-normalized 1536-D image embedding
```

The text encoder is not loaded on the EVK. Frozen text embeddings are committed in:

```text
benchmark_inputs/frozen_text_embeddings/
```

### 9.2 Why the ONNX graph was patched

The original exported visual graph contained constant cubic operations of the form:

```text
Pow(x, 3)
```

These were rewritten as:

```text
Mul(Mul(x, x), x)
```

using:

```text
model_build/siglip2/patch_siglip2_pow3.py
```

This change was required to obtain a deployable Qualcomm compilation path while preserving the intended computation.

### 9.3 Why the graph was split

A fused FP16 context was successfully compiled but could not be deployed on the EVK. FastRPC failed while mapping a persistent-weight buffer of:

```text
2,336,227,328 bytes
```

This corresponds to an approximately 2.45 GB linked context. The graph was therefore split at `add_1817`:

```text
Stage 1 output: [1, 576, 1536]
Stage 2 input:  [1, 576, 1536]
Stage 2 output: normalized 1536-D image embedding
```

The split is a memory-fit workaround, not a speed optimization. It introduces a host-visible intermediate tensor and a second QNN dispatch on the critical path.

### 9.4 Exact FP16 build sequence

1. Patch every constant `Pow(x, 3)` operation.
2. Split the patched model at `add_1817`.
3. Compile and link both fixed-batch stages for IQ-9075 HTP with QAIRT 2.45.
4. Use `FLOAT16`, `ENABLE_DLBC_WEIGHTS=1`, optimization level 3, and VTCM 0.
5. Pair each linked QNN context with an ONNX Runtime EPContext wrapper.
6. Validate embedding similarity and retrieval behavior.
7. Run through ONNX Runtime QNN with CPU fallback disabled.

Commands for constructing controlled candidates from an exported visual ONNX graph:

```bash
python3 model_build/siglip2/patch_siglip2_pow3.py \
  visual.onnx \
  build/visual_powfix.onnx

python3 model_build/siglip2/split_siglip2_visual_onnx.py \
  build/visual_powfix.onnx \
  build/siglip_split \
  --boundary add_1817

# Submit each validated stage to Qualcomm AI Hub, then compile/link it.
python3 model_build/siglip2/compile_siglip2_fp16_powfix_aihub.py \
  --source-model-id <ai_hub_stage_model_id> \
  --qairt-version 2.45 \
  --image-size 384 \
  --vtcm-mb 0 \
  --precision-label fp16 \
  --output-dir build/siglip_candidate
```

Recorded Qualcomm AI Hub source model IDs:

```text
Stage 1: mm64oz62q
Stage 2: mmd863okq
```

The complete compiler options, interfaces, job IDs, profiles, and hashes are committed in:

```text
config/model_artifacts/siglip2_split_fp16_manifest.json
```

### 9.5 Why W8A16 is not the baseline

W8A16 was investigated because completed generic QDQ candidates were smaller and faster. However, the tested candidates collapsed the image-embedding space and produced zero Top-1 retrieval agreement on the fixed validation gate.

Lite-MP attempts intended to preserve sensitive layers exceeded the Qualcomm AI Hub 90-minute job limit. The accuracy-safe baseline is therefore split FP16, not an INT8 visual encoder.

### 9.6 Qualcomm optimization target

The main SigLIP2 target is to remove or reduce the latency caused by:

- two QNN dispatches instead of one;
- the host-visible `[1, 576, 1536]` boundary transfer;
- the inability to deploy the fused 2.45 GB context;
- the lack of an accuracy-preserving lower-precision artifact.

Useful directions include a deployable fused context, improved persistent-weight mapping, a memory-aware graph partition that minimizes boundary cost, or mixed precision that preserves embedding cosine similarity and Top-K retrieval.

Any candidate must pass embedding cosine, Top-K retrieval, Task 1 F1, and Task 2 accuracy gates before its latency is compared with the baseline.

## 10. Qwen3-VL-Reranker-2B

### 10.1 Why a custom route is required

The selected model is an F16 **rank-pooling** GGUF converted from Qwen3-VL-Reranker-2B. It is not a standard causal-generation GGUF. It must preserve:

```text
cls.output.weight             two-row binary classifier
rank-pooling metadata
FP16 multimodal projector     separate mmproj GGUF
```

Ordinary causal-generation vocabulary logits do not reproduce the required relevance score. The pipeline therefore uses an MTMD/llama.cpp bridge:

- MTMD performs image encoding and multimodal projection;
- llama.cpp performs rank pooling;
- the bridge returns the binary relevance score used to rerank captions.

### 10.2 Exact Q8_0 conversion

The conversion is implemented in:

```text
model_build/qwen3_vl_reranker/quantize.sh
```

It uses llama.cpp commit:

```text
4f31eedb0ccf546b7e8d6bb243b170f12522f54d
```

Example command:

```bash
export LLAMA_QUANTIZE=/path/to/llama.cpp/build/bin/llama-quantize

./model_build/qwen3_vl_reranker/quantize.sh \
  /path/to/Qwen3-VL-Reranker-2B-F16-rank-pooling.gguf \
  external_assets/models/Qwen3-VL-Reranker-2B-GGUF/Qwen3-VL-Reranker-2B-Q8_0-F16Emb-F16Cls-HTP.gguf \
  "$LLAMA_QUANTIZE"
```

The wrapper:

- quantizes eligible transformer tensors to `Q8_0`;
- uses eight quantization threads;
- forces the token embedding table to FP16;
- forces `cls.output.weight` to FP16;
- keeps the separate multimodal projector in FP16.

Artifact sizes:

```text
F16 text GGUF: 3,447,362,112 bytes
Q8 text GGUF:  2,126,156,352 bytes
Total bundle reduction with FP16 projector retained: 31.0%
```

The score-sensitive embedding table, classifier, and multimodal boundary remain FP16 to limit ranking drift.

### 10.3 Why Q8_0 was selected

Paired 40-image screening produced:

| Text precision | Caption accuracy | Difference from F16 |
| --- | ---: | ---: |
| F16 | 95.0% | reference |
| Q8_0 | 92.5% | -2.5 percentage points |
| Q4 | 87.5% | -7.5 percentage points |

Q4 introduced unacceptable score drift. Q8_0 provided the best tested compromise between size and accuracy.

In the completed 350-image baseline, Q8_0 produced:

```text
Completed rows:       350 / 350
Failures/restarts:    0
Caption accuracy:     0.677143
Class accuracy:       0.877143
Mean latency:         7.97 seconds
```

### 10.4 Alternatives that were rejected

The following routes were tested but are not accuracy-equivalent replacements:

| Candidate | Outcome |
| --- | --- |
| Qwen vision W8A16 | unacceptable visual-feature drift |
| Full-graph W4A16 | ranking collapse |
| Full-graph W8A16 | AI Hub worker killed by out-of-memory condition |
| Four-stage W8A16 text deployment | 4.35 s projected latency, but only 56.0% caption accuracy |

### 10.5 Current heterogeneous execution

The selected runtime is hybrid:

```text
Qwen text/rank-pooling:   Qualcomm hybrid/HTP-capable llama.cpp route
Vision/projector:         MTMD on Adreno OpenCL
vision.UPSCALE:           CPU fallback
```

This route is functional and reproducible, but it introduces cross-backend synchronization and retains a CPU fallback.

### 10.6 Qualcomm optimization target

The reranker target is to reduce the 7.97-second mean latency while preserving rank-pooling semantics and the reference accuracy. Particularly useful improvements would be:

- native HTP support for the full rank-pooling model;
- an optimized multimodal projector and image encoder;
- elimination of `vision.UPSCALE` CPU fallback;
- fewer CPU/GPU/HTP transitions and synchronizations;
- a mixed-precision scheme that preserves classifier scores;
- a supported QAIRT deployment that exposes the two-row classifier rather than only causal logits.

A speedup that changes ranking behavior or loses Task 2 accuracy is not a valid replacement.

---

# Part 2B — Qwen3-VL-4B-Instruct Task 1 accuracy regression

## 11. Problem statement

Task 1 remains materially less accurate on the Dragonwing IQ-9075 EVK than on the NVIDIA Jetson AGX Orin, even though the current EVK result is now fully reproducible with the default benchmark command.

The comparison uses the same first 350 images, the same 626 ground-truth ingredient instances, the same Top-20 candidate policy, and the same fusion and final-selection configuration.

| Platform / run | Qwen deployment | F1 | Precision | Recall | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jetson AGX Orin | TensorRT Edge-LLM, INT4 weights / FP16 activations | 0.7579 | 0.7879 | 0.7300 | 457 | 123 | 169 |
| Dragonwing IQ-9075 EVK — current reproduced baseline | GenieX / QAIRT, W4A16 | 0.660054 | 0.757764 | 0.584665 | 366 | 117 | 260 |
| Dragonwing IQ-9075 EVK — archived 15-second-timeout diagnostic | GenieX / QAIRT, W4A16 | 0.6521 | 0.7558 | 0.5735 | 359 | 116 | 267 |

Relative to the Orin result, the **current reproduced EVK baseline** loses approximately:

```text
Absolute F1:            -0.097846
Relative F1:            -12.91%
Precision:              -0.030136
Recall:                 -0.145335
True positives:         -91
False negatives:        +91
```

The archived `0.6521` run is retained only as diagnostic evidence. Removing the old timeout behavior recovers seven true positives and raises EVK F1 to `0.660054`, but it does not close the main Orin-versus-EVK accuracy gap.

Two separately obtained GenieX-compatible Qwen3-VL-4B-Instruct W4A16 packages were tested and produced the same current EVK metrics. The observed gap is therefore not explained by choosing one of those two download/import routes.

## 12. Task 1 pipeline under investigation

Task 1 performs the following steps:

1. SigLIP2 retrieves the Top-20 candidate ingredients.
2. Qwen3-VL-4B-Instruct classifies each candidate as `present`, `possible`, or not visible.
3. Fusion combines normalized SigLIP2 retrieval scores and Qwen classifications.
4. The final selector emits the predicted ingredient set.

Principal diagnostic fusion configuration:

```text
visual_row_z_w0.25+qwen_present_max_row_z_w0.75
```

Important settings:

```text
Top-K candidates:             20
SigLIP2 fusion weight:        0.25
Qwen present fusion weight:   0.75
Qwen possible fusion weight:  0.0
Maximum generated tokens:     96
Qwen skip relative-gap rule:  0.25
```

Both full evaluations use the same first 350 images and the same 626 ground-truth ingredient instances.

### Orin VLM deployment

```text
Model:       Qwen3-VL-4B-Instruct
Runtime:     TensorRT Edge-LLM
Weights:     INT4
Activations: FP16
Mode:        persistent
```

Archived result:

```text
reports/orin_task1_legacy_fixed_top20.json
```

### Qualcomm EVK VLM deployment

```text
Model:             Qwen3-VL-4B-Instruct
Resolved artifact: local/qwen3vl-4b-qairt-w4a16
Runtime:           GenieX / QAIRT
Weights:           W4
Activations:       A16
Image size:        512 x 512
Padding:           white
```

Current reproduced result:

```text
run_outputs/task1_350/eval_first_350_samples.json
reference_results/task1_350/
```

Historical timeout diagnostic:

```text
reports/task1_evk_timeout15/eval_first_350_samples.json
```

## 13. Failure pattern and current interpretation

The primary final-metric difference is lost recall: the EVK emits fewer correct final ingredients and misses many labels recovered by the Orin pipeline.

The current no-timeout baseline confirms that request interruption was responsible for only a small part of the original loss. The archived timeout run contained:

```text
Timeout fallbacks:                    9
Rows with parser warnings:           17
Unsupported list-shaped payloads:     7
Invalid JSON payloads:                1
Rows with no parsed non-zero IDs:     19
Rows with 19 or 20 non-zero IDs:      77
Mean non-zero candidate count:     12.78
Median non-zero candidate count:      18
```

After the timeout behavior was removed, EVK F1 improved from `0.6521` to `0.660054`, while Orin remained at `0.7579`. This shows that timeout handling was a secondary issue rather than the root cause of the cross-platform gap.

The remaining diagnostic pattern is that successful EVK Qwen responses are frequently much broader than the corresponding Orin responses. Marking almost every candidate as non-zero can flatten relative ranking, destabilize row normalization, and interact poorly with the final threshold selector.

For rows where Qwen ran on both platforms in the archived paired trace, final micro-F1 was approximately `0.68` on Orin and `0.51` on EVK. When Qwen was skipped, the two pipelines were nearly equivalent. The major divergence is therefore concentrated in the Qwen-executed portion of Task 1.

## 14. Diagnostic Test 1 — Determine whether SigLIP2 explains the gap

The EVK SigLIP2 Top-20 retrieval result was:

```text
Top-20 recall:                                0.9201
Ground-truth labels present in Top-20:        576 / 626
Rows containing every truth label in Top-20: 317 / 350
```

A paired Orin/EVK trace comparison found:

```text
Same Top-1 label:             341 / 350 (97.4%)
Mean Top-20 label overlap:    18.96 / 20
Same Qwen skip/run decision:  338 / 350
Orin Qwen-run rows:           185
EVK Qwen-run rows:            179
Rows where both ran Qwen:     176
```

### Result

SigLIP2 does not explain the current approximately `0.098` F1 gap:

- 92% of all truth labels are available to the EVK Qwen stage;
- candidate sets are almost identical;
- Top-1 candidates match on more than 97% of rows;
- Qwen skip/run decisions match on more than 96% of rows;
- the large quality difference appears mainly after Qwen executes.

The identical-pixel experiment below also freezes the candidate lists exactly and reaches the same conclusion.

## 15. Diagnostic Test 2 — Inspect representative Qwen failures

### 15.1 `img_011027.jpg`

Ground truth:

```text
coffee
sweet bun
```

Orin Qwen response:

```json
{"present":[1,4,6],"possible":[10]}
```

Orin final selection:

```text
sweet bun
red bean paste
coffee
```

EVK Qwen response:

```json
{
  "present": [1,4,5,9,13,18],
  "possible": [2,3,6,7,8,10,11,12,14,15,16,17,19]
}
```

The EVK assigns a non-zero status to 19 of 20 candidates. Its final selection is:

```text
sweet bun
coffee
red bean paste
cream cheese frosting
breadstick
custard
```

This image shows the characteristic EVK failure: a visually narrow scene triggers an extremely broad classification.

### 15.2 `img_003871.jpg`

Ground truth:

```text
dal
roti
yogurt
```

The archived EVK run timed out and fell back to `roti`. A later completed deterministic call classified all 20 candidates as non-zero:

```json
{
  "present": [1,2,4,13,15,17,18,20],
  "possible": [3,5,6,7,8,9,10,11,12,14,16,19]
}
```

Fusion emitted:

```text
roti
dal
lentil
```

The corresponding Orin response was substantially more selective:

```json
{"present":[1,11,16],"possible":[2,4,13,14,15,17,19]}
```

### Result

Completing the EVK request does not by itself remove the quality regression. Successful EVK inference can still produce saturated candidate lists.

## 16. Diagnostic Test 3 — Freeze generation parameters

A historical diagnostic request used explicit token limits for both API conventions and deterministic decoding:

```json
{
  "max_completion_tokens": 96,
  "max_tokens": 96,
  "temperature": -1.0,
  "stream": false
}
```

This block documents a controlled experiment; it is **not** an additional reproduction step. The current repository code already implements the validated default behavior and the normal Task 1 benchmark command reproduces `0.660054`.

### Result

The controlled request did not eliminate overprediction:

- `img_011027.jpg` remained saturated;
- `img_003871.jpg` completed but classified all 20 candidates as non-zero;
- the later 100-image controlled EVK run still had a median of 19 non-zero candidates.

Generation-parameter handling was therefore not the primary cause of the remaining accuracy loss.

## 17. Diagnostic Test 4 — Identical-pixel cross-platform comparison

This is the strongest controlled test in the investigation.

### 17.1 Test contract

One hundred rows were selected from images for which both historical pipelines executed Qwen. The two representative images above were explicitly included. For every row:

- candidate labels and candidate order were frozen;
- the complete Qwen prompt was frozen;
- Qwen was forced to run;
- the source image was resized once with Pillow LANCZOS;
- aspect ratio was preserved;
- the image was centered on a white 512 x 512 RGB canvas;
- the canonical image was saved as PNG;
- no intermediate JPEG was used;
- its SHA-256 checksum was recorded in the manifest;
- the exact same PNG bytes were supplied to both backends;
- caller-side backend preprocessing was disabled.

Manifest:

```text
reports/task1_qwen_same_pixel_100/same_pixel_manifest.json
```

Canonical PNGs are intentionally not versioned. Regenerate them from the Drive-delivered Task 1 images with the runner's `prepare` command; the manifest retains the source-image and canonical-image SHA-256 values for verification.

This test removes the following cross-platform variables:

```text
JPEG encoding
caller-side resize
caller-side padding
SigLIP2 candidate retrieval
candidate order
skip logic
prompt construction
```

It does not force the private internal image processor or runtime kernels of the two backends to be identical.

### 17.2 Results

| Metric | Orin Edge-LLM | EVK GenieX |
| --- | ---: | ---: |
| Rows | 100 | 100 |
| Successful calls | 100 | 100 |
| Mean latency | 0.969 s | 3.879 s |
| Valid JSON rows | 100 | 95 |
| Valid schema rows | 90 | 82 |
| Mean `present` count | 3.06 | 4.16 |
| Mean `possible` count | 2.80 | 9.77 |
| Mean total non-zero count | 5.86 | 13.93 |
| Median total non-zero count | 2 | 19 |
| Rows with 19 or 20 non-zero IDs | 14 | 53 |
| Empty non-zero rows | 0 | 5 |

Present-only candidate metrics:

| Metric | Orin Edge-LLM | EVK GenieX |
| --- | ---: | ---: |
| F1 | 0.6874 | 0.4800 |
| Precision | 0.5784 | 0.3606 |
| Recall | 0.8469 | 0.7177 |
| TP | 177 | 150 |
| FP | 129 | 266 |
| FN | 32 | 59 |

Present-plus-possible candidate metrics:

| Metric | Orin Edge-LLM | EVK GenieX |
| --- | ---: | ---: |
| F1 | 0.4805 | 0.2310 |
| Precision | 0.3259 | 0.1328 |
| Recall | 0.9139 | 0.8852 |
| TP | 191 | 185 |
| FP | 395 | 1208 |
| FN | 18 | 24 |

Paired output comparison:

```text
Identical raw response:                6 / 100
Identical present set:                34 / 100
Identical possible set:               15 / 100
Mean EVK-minus-Orin non-zero count:      +8.07
Median EVK-minus-Orin non-zero count:   +10
```

### Result

JPEG recompression and caller-side resizing/padding are ruled out as the cause of the EVK regression.

With identical image bytes, prompts, and candidate lists, the EVK deployment remains broader, less precise, less accurate, less schema-compliant, and approximately four times slower. The large increase in `possible` predictions shows that the EVK artifact does not reliably distinguish weak visual evidence from absent evidence.

This experiment establishes a backend/model-artifact difference rather than a dataset or pipeline-input difference.

## 18. Diagnostic Test 5 — Correct, shuffled, and blank images

This test checks whether the EVK model actually uses its visual input or instead follows only candidate order and prompt patterns.

For each target row, the original prompt and candidate list were retained while the image was changed among:

1. the correct canonical image;
2. a different food image, shuffled with a fixed 37-row offset and no fixed points;
3. the same blank white 512 x 512 RGB PNG for every row.

### 18.1 Results

| EVK condition | Mean non-zero | Median non-zero | 19–20 rows | Empty rows | Present-only F1 | Present recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct image | 13.93 | 19 | 53 | 5 | 0.4800 | 0.7177 |
| Shuffled image | 4.96 | 3 | 13 | 34 | 0.1897 | 0.1770 |
| Blank image | 4.44 | 0 | 14 | 72 | 0.1598 | 0.1866 |

Paired set changes:

```text
Correct versus shuffled:
  Rows whose non-zero set changed:  91 / 100
  Mean non-zero Jaccard:             0.232
  Mean present Jaccard:              0.204

Correct versus blank:
  Rows whose non-zero set changed:  91 / 100
  Mean non-zero Jaccard:             0.217
  Mean present Jaccard:              0.138
```

The shuffled- and blank-image F1 values are diagnostic only. They compare predictions against the original row ground truth even though the supplied image is intentionally no longer the correct image for that row.

### Result

Replacing the image changed the predicted non-zero set on 91% of rows. Present recall fell from `0.7177` to approximately `0.18`, and the blank image produced an empty output on 72 rows.

Therefore, the Qwen visual input is neither disconnected nor completely ignored. The EVK deployment responds materially to the image.

## 19. Consolidated conclusion

The retained tests establish that:

1. the current Qualcomm EVK Task 1 baseline (`0.660054`) is substantially worse than the Orin result (`0.7579`);
2. removing the old timeout behavior makes the EVK baseline fully reproducible and improves it over the archived `0.6521` run, but does not close the cross-platform gap;
3. two separately obtained GenieX-compatible Qwen3-VL-4B-Instruct W4A16 packages produce the same EVK result;
4. the regression is concentrated after the Qwen stage, not in SigLIP2 retrieval;
5. successful EVK calls frequently classify 19 or 20 of 20 candidates as non-zero;
6. identical preprocessed image bytes, prompts, and candidate lists do not make EVK and Orin agree;
7. JPEG recompression, caller-side resize, padding, candidate retrieval, candidate order, skip logic, and prompt construction are not the primary cross-platform cause;
8. the visual path is active and the image is not ignored;
9. the EVK output is much more sensitive to prompt-example IDs and candidate positions;
10. the Orin TensorRT Edge-LLM deployment is substantially more selective and accurate under the same controlled inputs.

The practical problem is localized to the Qualcomm Qwen3-VL-4B-Instruct W4A16 GenieX/QAIRT deployment route, rather than to SigLIP2 or to the external image preparation performed by the pipeline.

The observed behavior is consistent with poor calibration or excessive numeric drift in the multimodal model:

- weak visual evidence is promoted to `possible`;
- related ingredients are not discriminated reliably;
- prompt-token patterns overpower visual selectivity;
- output logits enter an enumeration-like mode;
- response length and JSON/schema compliance become less stable.

### Important attribution limit

The evidence shows that the tested Qualcomm W4A16 deployment is not accuracy-equivalent to the Orin INT4/FP16 TensorRT Edge-LLM artifact. It strongly implicates the Qualcomm quantized deployment, but it does **not** isolate W4 weight quantization as the only possible cause.

Remaining internal causes include:

- AI Hub conversion and calibration recipe;
- quantization groupings, scales, clipping, and mixed-precision choices;
- multimodal projector conversion;
- image-token integration;
- Qwen3-VL chat template and special tokens;
- image-grid metadata and multimodal RoPE positions;
- attention masks;
- runtime kernels;
- decoding implementation.

No unquantized BF16/FP16 Qwen3-VL-4B reference was run with the exact same-pixel manifest. Such a reference is required to separate pure quantization loss from conversion, multimodal integration, and runtime effects.

The safe conclusion is:

> The tested Qualcomm QAIRT W4A16 Qwen3-VL-4B-Instruct deployment, as executed through GenieX on the IQ-9075 EVK, has a reproducible Task 1 accuracy and output-calibration regression relative to the tested TensorRT Edge-LLM INT4/FP16 deployment on Jetson AGX Orin. Vendor-level instrumentation is required to determine whether the root cause is quantization, calibration, conversion, multimodal integration, runtime execution, or a combination of these factors.

## 20. Accuracy-investigation artifacts

### Current full 350-image comparison

```text
Orin result:
reports/orin_task1_legacy_fixed_top20.json

EVK reproduced baseline:
run_outputs/task1_350/eval_first_350_samples.json
reference_results/task1_350/
```

### Historical timeout diagnostic

```text
EVK result:
reports/task1_evk_timeout15/eval_first_350_samples.json

EVK prediction table:
reports/task1_evk_timeout15/eval_first_350_samples_predictions.csv
```

### Deterministic single-image runs

```text
reports/task1_evk_greedy96/img_011027.json
reports/task1_evk_greedy96/img_003871.json
```

### Controlled same-pixel experiment

```text
Test runner:
reports/task1_qwen_same_pixel_100/run_same_pixel_qwen.py

Frozen manifest:
reports/task1_qwen_same_pixel_100/same_pixel_manifest.json

EVK raw result:
reports/task1_qwen_same_pixel_100/evk_geniex_results.json

Orin raw result:
reports/task1_qwen_same_pixel_100/orin_edgellm_results.json

Paired comparison:
reports/task1_qwen_same_pixel_100/orin_vs_evk_comparison.json
```

### Visual-conditioning controls

```text
Shuffled-image result:
reports/task1_qwen_same_pixel_100/visual_conditioning/evk_shuffled_image_results.json

Blank-image result:
reports/task1_qwen_same_pixel_100/visual_conditioning/evk_blank_image_results.json

Correct versus shuffled:
reports/task1_qwen_same_pixel_100/visual_conditioning/correct_vs_shuffled_comparison.json

Correct versus blank:
reports/task1_qwen_same_pixel_100/visual_conditioning/correct_vs_blank_comparison.json
```

---

# Final optimization acceptance rule

Qualcomm-side changes should be evaluated as replacements only when they improve latency without violating the immutable benchmark contract.

A candidate should be reported with:

```text
Artifact identity and SHA-256
Build/conversion/calibration configuration
Runtime and library versions
Processor placement and fallback map
Per-stage and end-to-end latency
Peak and incremental memory
Task 1 F1 / precision / recall
Task 2 caption / class accuracy
Candidate recall and guarded-row counts
Failure, parser, schema, and restart statistics
```

The desired outcome is to approach the Jetson AGX Orin accuracy and latency while maintaining reproducibility and making every implementation difference explicit.
