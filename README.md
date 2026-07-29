# Dishcovery on Qualcomm Dragonwing IQ-9075 EVK

## Reproduction baseline and optimization targets

This repository is the fixed Qualcomm baseline for the Dishcovery Task 1 and Task 2 pipelines. It has three purposes:

1. provide everything required to reproduce the current 350-image Qualcomm results;
2. document the current latency bottlenecks in SigLIP2 and Qwen3-VL-Reranker-2B, including how the deployed artifacts were produced and why those implementations were selected;
3. document the Task 1 accuracy regression observed with the Qualcomm Qwen3-VL-4B-Instruct W4A16 deployment, together with the controlled tests that localize the problem.

The goal is to let Qualcomm engineers reproduce the baseline without changing its configuration, then investigate faster and more accurate replacement implementations. The target is to approach the accuracy/F1 and latency obtained on NVIDIA Jetson AGX Orin while preserving the benchmark contract defined below.

> **Baseline rule:** do not alter the fixed commands, inputs, candidate policies, prompts, thresholds, or evaluation code when validating a replacement model. First reproduce the reference result; then change one implementation variable at a time.

## Contents

- [Part 1 — Reproduce the current results](#part-1--reproduce-the-current-results)
- [Part 2A — Known latency problems and model-build rationale](#part-2a--known-latency-problems-and-model-build-rationale)
- [Part 2B — Qwen3-VL-4B-Instruct Task 1 accuracy regression](#part-2b--qwen3-vl-4b-instruct-task-1-accuracy-regression)

---

# Part 1 — Reproduce the current results

## 1. Repository and asset layout

Git contains all source code, small inputs, cached text embeddings, configuration files, logs, and reference results. Google Drive supplies the 350-image data and the fixed SigLIP2 and Qwen3-VL-Reranker-2B binaries. Qwen3-VL-4B-Instruct is downloaded directly from Qualcomm AI Hub through GenieX and is never supplied through Google Drive.

Download the private asset package from [Google Drive](https://drive.google.com/drive/folders/1gGnXaYtdx4e8cTYwCkAXPakYQqqiBwc5?usp=drive_link), then extract it at the repository root so that it creates exactly this layout:

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

Verify all versioned inputs and Drive-delivered model artifacts:

```bash
python3 scripts/verify_external_assets.py
```

Every entry must be reported as `OK`. The expected SHA-256 values are committed in:

```text
config/checksums/model_and_input_sha256.txt
```

Qwen3-VL-4B-Instruct is intentionally absent from this checksum file because GenieX retrieves and manages it directly from Qualcomm AI Hub.

## 2. Repository map

```text
benchmark_inputs/       Versioned lists, labels, captions, mappings, and frozen text embeddings
config/                 Checksums, platform contract, model metadata, and benchmark settings
external_assets/        Ignored Drive extraction target: images plus SigLIP2/reranker binaries
model_build/            SigLIP2 graph/compile utilities and reranker MTMD/Q8 build code
pipeline/               Task 1 and Task 2 evaluation code; calorie support used by the demo
demo_web/               Browser demo server, backend, and static UI
reference_results/      Archived summaries, JSON, CSV, checkpoints, and logs
reports/                Accuracy-diagnostic manifests, raw outputs, and comparisons
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
onnxruntime-qnn:           2.1.0 / QAIRT 2.45.40
```

Do not mix QAIRT 2.47 ONNX Runtime/QNN libraries with the GenieX 2.45 runtime used by this baseline.

Create the Python environment:

```bash
python3 -m venv .venv_ort_qnn_245
.venv_ort_qnn_245/bin/python -m pip install --upgrade pip
.venv_ort_qnn_245/bin/python -m pip install -r requirements.txt

# Install the Qualcomm-provided ONNX Runtime 1.24.4 and
# ORT-QNN 2.1.0 / QAIRT 2.45.40 wheels into this environment.

export TASK1_PYTHON="$PWD/.venv_ort_qnn_245/bin/python"
export TASK2_PYTHON="$PWD/.venv_ort_qnn_245/bin/python"
export GENIEX_QNN_BACKEND="$HOME/.local/share/geniex/qairt/htp-files/libQnnHtp.so"
export DISHCOVERY_MODELS_DIR="$PWD/external_assets/models"
```

The fixed platform policy is recorded in:

```text
config/platform/evk_qairt_stack.json
```

The Task 2 runner validates this contract strictly.

## 4. Pull and start Qwen3-VL-4B-Instruct

After verifying the Drive assets, obtain the exact Task 1 VLM from Qualcomm AI Hub:

```bash
# Authenticate with Qualcomm AI Hub first if GenieX requests it.
geniex pull qwen3_vl_4b_instruct:w4a16 --model-hub aihub --model-type vlm

# Fixed model selection used by the benchmark runner.
export GENIEX_TASK1_MODEL="qualcomm/qwen3_vl_4b_instruct:w4a16"

geniex list
geniex serve --host 127.0.0.1:18181
```

`geniex list` must contain:

```text
qualcomm/qwen3_vl_4b_instruct:w4a16
```

Do not set `GENIEX_DATADIR` for this model and do not copy a Qwen binary into `external_assets/`. The pull command manages the AI Hub deployment. The exact model identity and command are recorded in:

```text
config/model_artifacts/task1_qwen3_vl_aihub_w4a16.json
```

The archived diagnostic reports refer to the locally resolved artifact as:

```text
local/qwen3vl-4b-qairt-w4a16
```

The reproduction procedure must nevertheless use the pinned AI Hub selection shown above.

## 5. Build the Qwen3-VL-Reranker-2B MTMD/llama.cpp bridge

The reference reranker is not executed through ordinary causal-generation logits. It is a rank-pooling model whose two-output classifier must be preserved. The native bridge combines MTMD multimodal processing with llama.cpp rank pooling.

Build it against the recorded llama.cpp revision and ABI-compatible GenieX libraries:

```bash
export LLAMA_CPP_SOURCE=/path/to/llama.cpp-at-4f31eedb0ccf546b7e8d6bb243b170f12522f54d
export GENIEX_LLAMA_LIB_DIR="$HOME/.local/share/geniex/llama_cpp"

./model_build/qwen3_vl_reranker/build.sh
```

At runtime:

- MTMD executes image encoding and the multimodal projector;
- llama.cpp executes rank pooling and returns the binary relevance score;
- the text model uses the Qualcomm hybrid/HTP-capable route;
- the FP16 vision/projector path uses the exposed MTMD Adreno OpenCL route;
- `vision.UPSCALE` has one CPU fallback.

## 6. Run the exact 350-image benchmarks

Keep the GenieX server running and execute the fixed runners without changing their arguments.

```bash
# Task 1: first 350 images, seed 7, legacy fixed Top-20 policy.
python3 scripts/run_350_benchmarks.py task1 \
  --qnn-backend-path "$GENIEX_QNN_BACKEND"

python3 scripts/verify_metrics.py task1 \
  run_outputs/task1_350/eval_first_350_samples.json

# Task 2: first 350 images, seed 42, class_caption evaluation,
# SigLIP Top-5, guarded Q8 reranker Top-5, gap 3.0,
# and five independent reranker calls.
python3 scripts/run_350_benchmarks.py task2 \
  --qnn-backend-path "$GENIEX_QNN_BACKEND"

python3 scripts/verify_metrics.py task2 \
  run_outputs/task2_350/eval_first_350_caption_alignment.json
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
- failures, malformed outputs, timeouts, and restarts.

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

The official Qualcomm AI Hub Qwen3-VL-4B-Instruct W4A16 artifact, executed through GenieX/QAIRT on the Dragonwing IQ-9075 EVK, has a reproducible quality and output-calibration regression on Task 1 ingredient recognition.

The investigation compares the same pipeline, dataset, ground truth, Top-20 candidate size, and fusion configuration across:

| Platform | Qwen deployment | F1 | Precision | Recall | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jetson AGX Orin | TensorRT Edge-LLM, INT4 weights / FP16 activations | 0.7579 | 0.7879 | 0.7300 | 457 | 123 | 169 |
| Dragonwing IQ-9075 EVK | Qualcomm AI Hub QAIRT W4A16 through GenieX | 0.6521 | 0.7558 | 0.5735 | 359 | 116 | 267 |

The archived EVK diagnostic run loses approximately:

```text
Absolute F1:            -0.1057
Relative F1:            approximately -14%
Recall:                 -0.1565
True positives:         -98
False negatives:        +98
```

The current immutable repository runner expects Task 1 F1 `0.660054`. The `0.6521` value above belongs to the archived diagnostic run in `reports/task1_evk_timeout15/`. These are separate recorded runs and should not be treated as conflicting benchmark expectations.

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
Resolved artifact: local/qwen3vl-4b-qairt-w4a16
Runtime:           GenieX / QAIRT
Weights:           W4
Activations:       A16
Image size:        512 x 512
Padding:           white
Timeout:           15 seconds in the archived full run
```

Archived result:

```text
reports/task1_evk_timeout15/eval_first_350_samples.json
```

## 13. Failure pattern in the full EVK run

The primary final-metric failure is lost recall: the EVK emits fewer final ingredients and misses many labels recovered by the Orin pipeline.

Qwen executed on 179 of the 350 EVK rows. Among those calls:

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

Raw Qwen responses are often excessively broad, while timeout or empty fallbacks remove Qwen evidence entirely. Both behaviors damage ranking and fusion. A response that marks almost every candidate as non-zero does not necessarily produce more final labels: it can flatten relative ranking, destabilize row normalization, and interact poorly with the final threshold selector.

For rows where Qwen ran on both platforms, paired final micro-F1 was approximately `0.68` on Orin and `0.51` on EVK. When Qwen was skipped, the two pipelines were nearly equivalent. The major divergence is therefore concentrated in the Qwen-executed portion of Task 1.

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

SigLIP2 does not explain the approximately 0.106 F1 gap:

- 92% of all truth labels are available to the EVK Qwen stage;
- candidate sets are almost identical;
- Top-1 candidates match on more than 97% of rows;
- Qwen skip/run decisions match on more than 96% of rows;
- the large quality difference appears mainly after Qwen executes.

The later identical-pixel experiment also freezes the candidate lists exactly and reaches the same conclusion.

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

The archived EVK run timed out, stored `{}`, and fell back to `roti`. With the corrected greedy 96-token request used in the next test, the call completed but classified all 20 candidates as non-zero:

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

Preventing a timeout does not remove the quality regression. Successful EVK inference can still produce saturated candidate lists.

## 16. Diagnostic Test 3 — Correct and freeze GenieX generation parameters

The diagnostic server request was corrected to use explicit token limits for both API conventions and deterministic greedy decoding:

```json
{
  "max_completion_tokens": 96,
  "max_tokens": 96,
  "temperature": -1.0,
  "stream": false
}
```

### Result

The corrected request did not eliminate overprediction:

- `img_011027.jpg` remained saturated;
- `img_003871.jpg` no longer timed out but classified all 20 candidates as non-zero;
- the later 100-image controlled EVK run still had a median of 19 non-zero candidates.

The original token/request configuration was therefore not the primary cause of the accuracy loss.

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

### 18.2 The visual path is active

Replacing the image changed the predicted non-zero set on 91% of rows. Present recall fell from `0.7177` to approximately `0.18`, and the blank image produced an empty output on 72 rows.

Therefore, the Qwen visual input is neither disconnected nor completely ignored. The EVK deployment responds materially to the image.

### 18.3 Prompt and candidate-position bias remain visible

Among the 28 non-empty blank-image responses:

```text
Responses containing ID 1:       28 / 28
Responses containing ID 4:       28 / 28
Responses containing ID 2:       27 / 28
Exact copies of [1,4] / [2]:       5
Responses with 19 or 20 IDs:      14
```

Those three IDs are the IDs used in the prompt's concrete JSON example:

```json
{"present":[1,4],"possible":[2]}
```

Some blank-image responses copied the example exactly. Others started with the same IDs and continued enumerating most remaining candidates.

### Result

Two conclusions hold simultaneously:

1. the EVK model genuinely uses visual information;
2. its outputs are strongly contaminated by prompt-example and candidate-position bias.

Real food images frequently push the model into a dense enumeration mode:

```text
Saturated correct-image rows: 53
Saturated shuffled rows:      13
Saturated blank rows:         14
```

A plausible behavioral interpretation is that the prompt creates a strong numeric continuation pattern, while the Qualcomm artifact insufficiently separates visually related candidates. Greedy decoding then expands the pattern into long enumerations, increasing malformed JSON, token-limit, and timeout failures.

## 19. Consolidated conclusion

The retained tests establish that:

1. the Qualcomm EVK Task 1 result is substantially worse than the Orin result;
2. the regression is concentrated after the Qwen stage, not in SigLIP2 retrieval;
3. the corrected deterministic GenieX request does not solve the problem;
4. successful EVK calls frequently classify 19 or 20 of 20 candidates as non-zero;
5. identical preprocessed image bytes, prompts, and candidate lists do not make EVK and Orin agree;
6. JPEG recompression, caller-side resize, padding, candidate retrieval, candidate order, skip logic, and prompt construction are not the primary cross-platform cause;
7. the visual path is active and the image is not ignored;
8. the EVK output is much more sensitive to prompt-example IDs and candidate positions;
9. the Orin TensorRT Edge-LLM deployment is substantially more selective and accurate under the same controlled inputs.

The practical problem is localized to the Qualcomm Qwen3-VL-4B-Instruct W4A16 QAIRT/GenieX artifact and deployment stack:

```text
local/qwen3vl-4b-qairt-w4a16
```

The observed behavior is consistent with poor calibration or excessive numeric drift in the multimodal model:

- weak visual evidence is promoted to `possible`;
- related ingredients are not discriminated reliably;
- prompt-token patterns overpower visual selectivity;
- output logits enter an enumeration-like mode;
- response length, JSON formatting, and timeout behavior become unstable.

### Important attribution limit

The evidence shows that the tested Qualcomm W4A16 artifact is not accuracy-equivalent to the Orin INT4/FP16 TensorRT Edge-LLM artifact. It strongly implicates the Qualcomm quantized deployment, but it does **not** isolate W4 weight quantization as the only possible cause.

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

> The official Qualcomm AI Hub QAIRT W4A16 Qwen3-VL-4B-Instruct artifact, as executed through GenieX on the IQ-9075 EVK, has a reproducible Task 1 accuracy and output-calibration regression. It is materially worse than the tested TensorRT Edge-LLM INT4/FP16 deployment on Jetson AGX Orin. Vendor-level instrumentation is required to determine whether the root cause is quantization, calibration, conversion, multimodal integration, runtime execution, or a combination of these factors.

## 20. Accuracy-investigation artifacts

### Full 350-image comparison

```text
Orin result:
reports/orin_task1_legacy_fixed_top20.json

EVK result:
reports/task1_evk_timeout15/eval_first_350_samples.json

EVK prediction table:
reports/task1_evk_timeout15/eval_first_350_samples_predictions.csv
```

### Corrected deterministic single-image runs

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
Timeout, parser, schema, and restart statistics
```

The desired outcome is to approach the Jetson AGX Orin accuracy and latency while maintaining reproducibility and making every implementation difference explicit.
