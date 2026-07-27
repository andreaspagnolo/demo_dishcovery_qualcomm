# Dishcovery: Qualcomm 350-image reproduction baseline

This repository is the immutable baseline for Qualcomm optimization work. It contains one fixed configuration for Task 1 and Task 2, the current browser demo defaults, and complete reference outputs. Do not alter the commands below when benchmarking a replacement model.

## Quick path: reproduce the published results

Git contains all source code, small inputs, cached text embeddings, configuration, logs, and reference results. Google Drive supplies the 350-image data and only the fixed SigLIP and Qwen3-VL-Reranker binaries. Qwen3-VL-Instruct-4B is downloaded directly from Qualcomm AI Hub with GenieX, never from Drive. Extract the Drive package exactly here:

```text
external_assets/
├── images/
│   ├── task1_350/                         # 350 img_*.jpg files
│   ├── task2_350/                         # Food-500 class/image hierarchy
│   └── demo/                              # browser gallery, up to 80 images
└── models/
    ├── siglip2_qcs9075_out/fp16_powfix_split_qairt245/
    │   ├── stage1/model.onnx
    │   ├── stage1/model.bin
    │   ├── stage2/model.onnx
    │   └── stage2/model.bin
    └── Qwen3-VL-Reranker-2B-GGUF/
        ├── Qwen3-VL-Reranker-2B-Q8_0-F16Emb-F16Cls-HTP.gguf
        └── mmproj-Qwen3-VL-Reranker-2B-F16.gguf
```

Download the private asset package from [Google Drive](https://drive.google.com/drive/folders/1gGnXaYtdx4e8cTYwCkAXPakYQqqiBwc5?usp=drive_link), then extract it at the repository root so it creates the `external_assets/` layout above. Then verify the versioned inputs and Drive-delivered SigLIP/reranker files:

```bash
python3 scripts/verify_external_assets.py
```

Every entry must be `OK`. The exact hashes are in `config/checksums/model_and_input_sha256.txt`. Qwen is intentionally absent from this checksum file because it is retrieved directly from Qualcomm AI Hub.

## Repository map

```text
benchmark_inputs/       Versioned lists, labels, captions, mappings, and frozen text embeddings
config/                 Checksums, platform contract, and model metadata
external_assets/        Ignored Google Drive extraction target: 350 images plus SigLIP and reranker binaries
model_build/            SigLIP graph/compile utilities and Qwen MTMD/Q8 build code
pipeline/               Task 1 and Task 2 evaluation code; calorie support used by the current demo
demo_web/               Browser demo server, backend, and static UI
reference_results/      Results summary and complete archived JSON, CSV, checkpoint, and logs
scripts/                Fixed run, verification, and demo entrypoints
```

## Platform and environment

Recorded platform: Qualcomm Dragonwing IQ-9075 EVK (`aarch64`), GenieX 0.3.13, QAIRT 2.45.0.260326, ONNX Runtime 1.24.4, and onnxruntime-qnn 2.1.0 / QAIRT 2.45.40. Do not mix QAIRT 2.47 ORT-QNN libraries with GenieX 2.45.

```bash
python3 -m venv .venv_ort_qnn_245
.venv_ort_qnn_245/bin/python -m pip install --upgrade pip
.venv_ort_qnn_245/bin/python -m pip install -r requirements.txt

# Install the Qualcomm-provided ONNX Runtime 1.24.4 and
# ORT-QNN 2.1.0/QAIRT 2.45.40 wheels into this same environment.
export TASK1_PYTHON="$PWD/.venv_ort_qnn_245/bin/python"
export TASK2_PYTHON="$PWD/.venv_ort_qnn_245/bin/python"
export GENIEX_QNN_BACKEND="$HOME/.local/share/geniex/qairt/htp-files/libQnnHtp.so"
export DISHCOVERY_MODELS_DIR="$PWD/external_assets/models"
```

The policy is in `config/platform/evk_qairt_stack.json`; the Task 2 command checks it strictly.

## Exact deployment-model procedure

First obtain the Drive package in the layout above and run `python3 scripts/verify_external_assets.py`. Its hashes guarantee that the QAIRT SigLIP contexts and Q8 reranker GGUF files are exactly the reference versions.

Then obtain Task 1 Qwen directly from Qualcomm AI Hub. This is the exact pinned deployment selection used by the fixed runner:

```bash
# Authenticate with Qualcomm AI Hub first if GenieX asks for it.
geniex pull qwen3_vl_4b_instruct:w4a16 --model-hub aihub --model-type vlm

# The pipeline uses this exact GenieX model name.
export GENIEX_TASK1_MODEL="qualcomm/qwen3_vl_4b_instruct:w4a16"
geniex list
geniex serve --host 127.0.0.1:18181
```

`geniex list` must include `qualcomm/qwen3_vl_4b_instruct:w4a16`. Do not set `GENIEX_DATADIR` for this model and do not copy a Qwen binary into `external_assets/`; the pull command manages the AI Hub deployment in GenieX. The exact command and identity are recorded in `config/model_artifacts/task1_qwen3_vl_aihub_w4a16.json`.

## How the deployed SigLIP and reranker were built

The Drive archive remains the source of the exact benchmark binaries: `python3 scripts/verify_external_assets.py` verifies their hashes. The build record below documents how those artifacts were produced and the exact constraints that led to their selected precision. It is intentionally separate from future optimization candidates.

### SigLIP2: split FP16 QNN contexts

The selected recall model is the visual encoder of `timm/ViT-gopt-16-SigLIP2-384`. Its fixed EVK contract is RGB bicubic resize to 384 x 384, NCHW `float32`, mean/std `[0.5, 0.5, 0.5]`, and one normalized 1536-dimensional image embedding. Text embeddings are frozen in `benchmark_inputs/frozen_text_embeddings/`, so the SigLIP text encoder is not loaded on the EVK.

The build sequence was:

1. Rewrite every constant `Pow(x, 3)` in the visual ONNX graph as `Mul(Mul(x, x), x)` using `model_build/siglip2/patch_siglip2_pow3.py`.
2. Split the patched visual graph at `add_1817` with `model_build/siglip2/split_siglip2_visual_onnx.py`. Stage 1 produces a host-visible `[1, 576, 1536]` boundary tensor; Stage 2 consumes it and produces the 1536-D embedding.
3. Compile and link both fixed-batch stages for Dragonwing IQ-9075 HTP with QAIRT 2.45 and `FLOAT16`, `ENABLE_DLBC_WEIGHTS=1`, optimization level 3, and VTCM 0. Each linked context is paired with an ONNX Runtime EPContext wrapper (`model.onnx`) and its QNN context (`model.bin`).
4. Validate the two-stage embedding/retrieval behavior, then use ONNX Runtime QNN with CPU fallback disabled.

The recorded Qualcomm AI Hub source model IDs are `mm64oz62q` (Stage 1) and `mmd863okq` (Stage 2); the compile, link, profile job IDs, compiler options, interfaces, and SHA-256 values are committed in `config/model_artifacts/siglip2_split_fp16_manifest.json`.

A fused FP16 context was also compiled, but it could not be deployed: FastRPC failed to map its 2,336,227,328-byte persistent-weight buffer (about a 2.45 GB context). Splitting is therefore a memory-fit decision, not a performance optimization; the intermediate transfer and second QNN dispatch remain on the critical path.

W8A16 was investigated but is deliberately not used. Two completed generic QDQ candidates fit and were faster, but collapsed the image-embedding space: zero top-1 retrieval agreement on the fixed validation gate. Lite-MP attempts that restored sensitive layers exceeded the AI Hub 90-minute limit. The safe baseline is consequently **split FP16**, not an INT8 SigLIP artifact.

The packaged utilities can construct controlled candidates from an exported visual ONNX graph:

```bash
python3 model_build/siglip2/patch_siglip2_pow3.py visual.onnx build/visual_powfix.onnx
python3 model_build/siglip2/split_siglip2_visual_onnx.py build/visual_powfix.onnx build/siglip_split --boundary add_1817

# Submit each validated stage to AI Hub, then compile/link a candidate with QAIRT 2.45.
python3 model_build/siglip2/compile_siglip2_fp16_powfix_aihub.py \
  --source-model-id <ai_hub_stage_model_id> \
  --qairt-version 2.45 --image-size 384 --vtcm-mb 0 \
  --precision-label fp16 --output-dir build/siglip_candidate
```

A candidate compilation is not a substitute for the hash-verified Drive contexts: it must pass embedding cosine, top-k retrieval, Task 1 F1, and Task 2 accuracy gates before comparison.

### Qwen3-VL-Reranker-2B: Q8_0 text with FP16 boundaries

The selected reranker starts from an F16 **rank-pooling** GGUF converted from Qwen3-VL-Reranker-2B. It is not a causal-generation GGUF: it must retain the two-row `cls.output.weight` classifier and rank-pooling metadata. The paired multimodal projector remains a separate FP16 `mmproj` GGUF.

The exact Q8 conversion is implemented in `model_build/qwen3_vl_reranker/quantize.sh` and uses the recorded llama.cpp commit `4f31eedb0ccf546b7e8d6bb243b170f12522f54d`:

```bash
export LLAMA_QUANTIZE=/path/to/llama.cpp/build/bin/llama-quantize

./model_build/qwen3_vl_reranker/quantize.sh \
  /path/to/Qwen3-VL-Reranker-2B-F16-rank-pooling.gguf \
  external_assets/models/Qwen3-VL-Reranker-2B-GGUF/Qwen3-VL-Reranker-2B-Q8_0-F16Emb-F16Cls-HTP.gguf \
  "$LLAMA_QUANTIZE"
```

The wrapper quantizes eligible transformer tensors with `Q8_0` and eight threads, while forcing the token embedding table and `cls.output.weight` to FP16. The text GGUF is 2,126,156,352 bytes compared with 3,447,362,112 bytes for F16. Keeping the projector FP16 gives a 31.0% total bundle-size reduction, while preserving the score-sensitive embedding, classifier, and multimodal boundaries.

This choice was evidence-driven. On the paired 40-image screen, Q8 reached 92.5% caption accuracy versus 95.0% for F16, a 2.5-point loss; Q4 lost 7.5 points and showed unacceptable score drift. In the completed 350-image run, Q8 finished 350/350 rows with zero failures or restarts, 0.677143 caption accuracy, 0.877143 class accuracy, and 7.97 seconds mean latency.

Other investigated paths are not drop-in replacements: Qwen vision W8A16 caused unacceptable feature drift; full-graph W4A16 caused ranking collapse; full-graph W8A16 was OOM-killed in the AI Hub worker; and a four-stage W8A16 text deployment achieved 4.35 seconds projected latency but only 56.0% caption accuracy.

At runtime, MTMD performs the FP16 vision encoder/projector work and llama.cpp performs rank pooling. The selected route is hybrid: Qwen text uses the Qualcomm HTP-capable path, the vision/projector uses Adreno OpenCL, and `vision.UPSCALE` has a CPU fallback. The native bridge must therefore be built against ABI-compatible llama.cpp and GenieX libraries.

## Build the MTMD/llama.cpp bridge

```bash
export LLAMA_CPP_SOURCE=/path/to/llama.cpp-at-4f31eedb0ccf546b7e8d6bb243b170f12522f54d
export GENIEX_LLAMA_LIB_DIR="$HOME/.local/share/geniex/llama_cpp"
./model_build/qwen3_vl_reranker/build.sh
```

MTMD performs multimodal image encoding/projector execution. llama.cpp performs rank pooling and returns the binary relevance score. Causal-generation vocabulary logits do not reproduce this score. The reference is hybrid: Qwen text uses the Qualcomm hybrid/HTP route; the FP16 vision/projector uses the exposed MTMD OpenCL route, with one UPSCALE CPU fallback.

## Exact 350-image runs

With the GenieX server running and the native bridge built:

```bash
# Task 1: first 350 images, seed 7, legacy fixed top-20 policy.
python3 scripts/run_350_benchmarks.py task1 --qnn-backend-path "$GENIEX_QNN_BACKEND"
python3 scripts/verify_metrics.py task1 run_outputs/task1_350/eval_first_350_samples.json

# Task 2: first 350 images, seed 42, class_caption, SigLIP top-5,
# guarded Q8 reranker top-5, keep gap 3.0, five individual reranker calls.
python3 scripts/run_350_benchmarks.py task2 --qnn-backend-path "$GENIEX_QNN_BACKEND"
python3 scripts/verify_metrics.py task2 run_outputs/task2_350/eval_first_350_caption_alignment.json
```

| Task | Metric | Expected |
| --- | --- | ---: |
| Task 1 | micro-F1 | 0.660054 |
| Task 1 | precision / recall | 0.757764 / 0.584665 |
| Task 2 | caption accuracy | 0.677143 (237/350) |
| Task 2 | class accuracy | 0.877143 (307/350) |

Task 1 locks first-350 ordering, seed 7, legacy SigLIP+Qwen logic, top-20 candidates, `present_possible`, 96 tokens, 512x512 white letterbox, visual/Qwen weights 0.25/0.75, global row-z `threshold_ratio` selection (3.5, 0.85, max 7), and a 0.25 VLM skip gap. Task 2 locks all candidate and reranker values listed above. Fresh logs and complete JSON/CSV outputs go to `run_outputs/`.

`reference_results/350_subset_summary.md` is the concise report. `reference_results/task1_350/` and `reference_results/task2_350/` contain the complete original evidence. Timing is reference evidence, not a pass/fail requirement on another power/thermal configuration.

## Browser demo

The browser demo retains its current default policy: Task 1 preload, SigLIP-guarded Q8 MTMD Task 2 when requested, and images from `external_assets/images/demo`.

```bash
python3 scripts/start_demo.py
# Open http://127.0.0.1:8787
```

## Optimization rule

Keep the commands above unchanged. For every faster candidate, record model hashes, GenieX/QAIRT/llama.cpp versions, HTP/GPU/CPU placement and fallbacks, latency, RSS, Task 1 F1, Task 2 caption/class accuracy, candidate recall, guarded-row count, and reranked-pair count. A speed gain is not a replacement until it stays within the stated metric tolerance.
