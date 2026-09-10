# Dishcovery on Qualcomm Dragonwing IQ-9075 EVK

Follow these steps to reproduce the two 350-image benchmarks and run the browser demo on a **Dragonwing IQ-9075 EVK running Qualcomm's Ubuntu 24.04 image (AArch64, Python 3.12)**.

| Benchmark | Model | Expected result |
| --- | --- | ---: |
| Task 1: ingredient recognition | Custom Qwen3-VL-4B-Instruct W8A16, INT8 KV cache | **0.706186 micro-F1** |
| Task 2: caption retrieval | Qwen3-VL-Reranker-2B Q8_0, FP16 embedding/projector, k=5 | **0.677143 caption accuracy** |

Both tasks use split FP16 SigLIP2 recall. The commands select the fixed image lists and inference settings automatically.

## 1. Install prerequisites and clone the repository

Run on the EVK as a regular user; use `sudo` only for system package installation:

```bash
sudo apt update
sudo apt install -y \
  python3.12 python3.12-venv python3-pip \
  git curl unzip cmake ninja-build build-essential pkg-config \
  clinfo qcom-adreno-cl1 qcom-fastrpc1

clinfo -l
```

`clinfo` must list `QUALCOMM Snapdragon(TM)` and `QUALCOMM Adreno(TM) 663`. Keep the Qualcomm OpenCL package; `ocl-icd-opencl-dev` conflicts with it.

```bash
git clone https://github.com/andreaspagnolo/demo_dishcovery_qualcomm.git
cd demo_dishcovery_qualcomm
```

Run every remaining command from this repository directory, using the same Linux user. When opening another terminal, open it in this directory. Run the code blocks in order; stop if a command fails.

## 2. Download the assets from Google Drive

Download these three files from the [shared Google Drive folder](https://drive.google.com/drive/folders/1gGnXaYtdx4e8cTYwCkAXPakYQqqiBwc5?usp=drive_link) into the repository directory:

- `dishcovery_350_assets_siglip_reranker.zip`
- `Qwen3-VL-4B-Instruct-GenieX-QAIRT-W8A16.zip`
- `Qwen3-VL-4B-Instruct-GenieX-QAIRT-W8A16.zip.sha256`

Extract and verify them:

```bash
(
set -e
sha256sum -c Qwen3-VL-4B-Instruct-GenieX-QAIRT-W8A16.zip.sha256
unzip -q dishcovery_350_assets_siglip_reranker.zip
mkdir -p external_assets/models
unzip -q Qwen3-VL-4B-Instruct-GenieX-QAIRT-W8A16.zip -d external_assets/models
python3.12 scripts/verify_external_assets.py
cd external_assets/models/Qwen3-VL-4B-Instruct-GenieX-QAIRT-W8A16
sha256sum -c SHA256SUMS
)
```

Every checksum must report `OK`. The resulting directories are:

```text
external_assets/
├── images/
│   ├── task1_350/
│   ├── task2_350/
│   └── demo/
└── models/
    ├── siglip2_qcs9075_out/fp16_powfix_split_qairt245/
    ├── Qwen3-VL-Reranker-2B-GGUF/
    └── Qwen3-VL-4B-Instruct-GenieX-QAIRT-W8A16/
```

Image lists, labels, caption banks, and cached text embeddings are included in Git. Software dependencies are downloaded by the commands below; no model compilation is required.

## 3. Install the Python and QNN runtime

The required runtime is **ONNX Runtime 1.24.4 + onnxruntime-qnn 2.1.0 / QAIRT 2.45.40**. Create the Python 3.12 environment:

```bash
(
set -e
python3.12 -m venv .venv_ort_qnn_245
.venv_ort_qnn_245/bin/python -m pip install --upgrade pip
.venv_ort_qnn_245/bin/python -m pip install -r requirements.txt
)
```

Install the exact AArch64 wheels, replacing any ONNX Runtime installed as a dependency:

```bash
(
set -e
WHEEL_DIR="external_assets/wheels"
ORT_WHEEL="onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
QNN_WHEEL="onnxruntime_qnn-2.1.0-cp312-cp312-manylinux_2_34_aarch64.whl"
ORT_URL="https://files.pythonhosted.org/packages/aa/60"
ORT_URL+="/c4d1c8043eb42f8a9aa9e931c8c293d289c48ff463267130eca97d13357f/$ORT_WHEEL"
QNN_URL="https://files.pythonhosted.org/packages/d8/71"
QNN_URL+="/17a145e8c4dc4a5de64ed3784382233845293445341bdb56c5d097827efd/$QNN_WHEEL"

mkdir -p "$WHEEL_DIR"
curl --fail --location --retry 3 -o "$WHEEL_DIR/$ORT_WHEEL" "$ORT_URL"
curl --fail --location --retry 3 -o "$WHEEL_DIR/$QNN_WHEEL" "$QNN_URL"
printf '%s  %s\n' \
  '1a5c5a544b22f90859c88617ecb30e161ee3349fcc73878854f43d77f00558b5' "$WHEEL_DIR/$ORT_WHEEL" \
  '5c8a3f5d95f05722f8b923212754fc76435b2132c64ae97478d569855c1ee2a8' "$WHEEL_DIR/$QNN_WHEEL" \
  | sha256sum --check

.venv_ort_qnn_245/bin/python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-qnn
.venv_ort_qnn_245/bin/python -m pip install --no-index --no-deps \
  "$WHEEL_DIR/$ORT_WHEEL" "$WHEEL_DIR/$QNN_WHEEL"
)
```

Check the installed versions and QNN provider:

```bash
.venv_ort_qnn_245/bin/python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import onnxruntime as ort
import onnxruntime_qnn as qnn

assert ort.__version__ == "1.24.4", ort.__version__
assert version("onnxruntime-qnn") == "2.1.0"
provider = qnn.get_ep_name()
if not any(device.ep_name == provider for device in ort.get_ep_devices()):
    ort.register_execution_provider_library(provider, qnn.get_library_path())
assert any(device.ep_name == provider for device in ort.get_ep_devices())
assert Path(qnn.get_qnn_htp_path()).is_file()
print("ONNX Runtime 1.24.4 / ORT-QNN 2.1.0: OK")
PY
```

## 4. Install GenieX and register the Task 1 model

Install the pinned **GenieX 0.3.13** release from Qualcomm's distribution. The installer verifies the runtime archive checksum.

```bash
(
set -e
curl --fail --location --retry 3 \
  https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/install-v0.3.13.sh \
  -o /tmp/dishcovery-geniex-install-v0.3.13.sh
printf '%s  %s\n' \
  'c570bb8df22b9180c49a2aad0f4a848ef8f114dacaad08f7381b5efee044771d' \
  '/tmp/dishcovery-geniex-install-v0.3.13.sh' | sha256sum --check
sh /tmp/dishcovery-geniex-install-v0.3.13.sh \
  --version v0.3.13 --prefix "$HOME/.local/share/geniex"
)

"$HOME/.local/bin/geniex" --skip-update --version
```

The output must show `GenieX CLI Version: v0.3.13` and `QAIRT Runtime Version: v2.45.0.260326`. Use this version throughout the guide.

Register the downloaded custom model once:

```bash
"$HOME/.local/bin/geniex" --skip-update pull local/qwen3vl-4b-qairt-w8a16 \
  --model-hub localfs \
  --local-path "$PWD/external_assets/models/Qwen3-VL-4B-Instruct-GenieX-QAIRT-W8A16" \
  --model-type vlm

"$HOME/.local/bin/geniex" --skip-update list
```

The list must include `local/qwen3vl-4b-qairt-w8a16`.

## 5. Build the Task 2 reranker bridge

Build the repository's native bridge against the pinned llama.cpp headers and the installed GenieX libraries:

```bash
(
set -e
export LLAMA_CPP_SOURCE="$PWD/third_party/llama.cpp"
export GENIEX_LLAMA_LIB_DIR="$HOME/.local/share/geniex/llama_cpp"
export QAIRT_HTP_DIR="$HOME/.local/share/geniex/qairt/htp-files"

mkdir -p third_party
if [ ! -d "$LLAMA_CPP_SOURCE/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_SOURCE"
fi
git -C "$LLAMA_CPP_SOURCE" fetch origin 4f31eedb0ccf546b7e8d6bb243b170f12522f54d
git -C "$LLAMA_CPP_SOURCE" checkout --detach 4f31eedb0ccf546b7e8d6bb243b170f12522f54d
./model_build/qwen3_vl_reranker/build.sh

export LD_LIBRARY_PATH="$GENIEX_LLAMA_LIB_DIR:$QAIRT_HTP_DIR"
export ADSP_LIBRARY_PATH="$GENIEX_LLAMA_LIB_DIR;$QAIRT_HTP_DIR"
export DSP_LIBRARY_PATH="$ADSP_LIBRARY_PATH"
RERANKER_BIN="$PWD/model_build/qwen3_vl_reranker/build/qwen3-vl-reranker-mtmd"
if ldd "$RERANKER_BIN" | grep -q 'not found'; then
  ldd "$RERANKER_BIN"
  exit 1
fi
DEVICE_LIST=$("$RERANKER_BIN" --list-devices)
printf '%s\n' "$DEVICE_LIST"
for backend in GPUOpenCL HTP0 CPU; do
  grep -Fq "$backend" <<<"$DEVICE_LIST"
done
)
```

The device list must include **GPUOpenCL**, **HTP0**, and **CPU**.

## 6. Run and verify the 350-image benchmarks

Run Task 1 first, stop its GenieX server, then run Task 2. Both terminal windows must be in the repository directory.

### Task 1

In **Terminal A**, start GenieX and leave it running:

```bash
unset LD_LIBRARY_PATH ADSP_LIBRARY_PATH DSP_LIBRARY_PATH
"$HOME/.local/bin/geniex" --skip-update serve --host 127.0.0.1:18181
```

Wait for `Local hosting on http://127.0.0.1:18181/`. In **Terminal B**, run and verify Task 1:

```bash
(
set -e
unset LD_LIBRARY_PATH ADSP_LIBRARY_PATH DSP_LIBRARY_PATH
export GENIEX_TASK1_MODEL="local/qwen3vl-4b-qairt-w8a16"
export DISHCOVERY_MODELS_DIR="$PWD/external_assets/models"
.venv_ort_qnn_245/bin/python scripts/run_350_benchmarks.py task1 \
  --qnn-backend-path "$HOME/.local/share/geniex/qairt/htp-files/libQnnHtp.so"
.venv_ort_qnn_245/bin/python scripts/verify_metrics.py task1 \
  run_outputs/task1_350/eval_first_350_samples.json
)
```

Expected verification values:

```text
f1: actual=0.706186 expected=0.706186 delta=+0.000000
precision: actual=0.763941 expected=0.763941 delta=+0.000000
recall: actual=0.656550 expected=0.656550 delta=+0.000000
```

### Task 2

When Task 1 finishes, press **Ctrl+C in Terminal A** to stop GenieX. Task 2 loads its own native runtime and requires the Task 1 server to be stopped.

In **Terminal B**:

```bash
(
set -e
if pgrep -af '[g]eniex.*serve'; then
  echo "Stop the GenieX server in Terminal A before continuing." >&2
  exit 1
fi
export DISHCOVERY_MODELS_DIR="$PWD/external_assets/models"
export GENIEX_LLAMA_LIB_DIR="$HOME/.local/share/geniex/llama_cpp"
export QAIRT_HTP_DIR="$HOME/.local/share/geniex/qairt/htp-files"
export LD_LIBRARY_PATH="$GENIEX_LLAMA_LIB_DIR:$QAIRT_HTP_DIR"
export ADSP_LIBRARY_PATH="$GENIEX_LLAMA_LIB_DIR;$QAIRT_HTP_DIR"
export DSP_LIBRARY_PATH="$ADSP_LIBRARY_PATH"
.venv_ort_qnn_245/bin/python scripts/run_350_benchmarks.py task2 \
  --qnn-backend-path "$QAIRT_HTP_DIR/libQnnHtp.so"
.venv_ort_qnn_245/bin/python scripts/verify_metrics.py task2 \
  run_outputs/task2_350/eval_first_350_caption_alignment.json
)
```

Expected verification values:

```text
top1_caption_accuracy: actual=0.677143 expected=0.677143 delta=+0.000000
class_top1_accuracy: actual=0.877143 expected=0.877143 delta=+0.000000
```

The verifier allows an absolute difference of 0.005. The frozen Task 1 settings use Top-20 ingredients and a 96-token response; Task 2 uses guarded Top-5 reranking with a SigLIP gap of 3.0.

The runners write progress to `run_outputs/task1_350/run.log` and `run_outputs/task2_350/run.log`. To watch the active task, open another terminal and use:

```bash
tail -f run_outputs/task1_350/run.log
```

For Task 2, use `run_outputs/task2_350/run.log`. Complete reports and predictions are saved alongside the logs. Recorded reference reports are in `reference_results/task1_350/` and `reference_results/task2_350/`; measured latency depends on device load and temperature.

## 7. Run the browser demo

The web app uses the same custom W8A16 Task 1 model and Q8 Task 2 reranker. Its voice input and output use Qualcomm Whisper-Base and PiperTTS-EN.

Download the public speech models once, using a separate downloader environment:

```bash
(
set -e
python3.12 -m venv .venv_qaihm
.venv_qaihm/bin/python -m pip install --upgrade pip
.venv_qaihm/bin/python -m pip install "qai-hub-models==0.58.0"
.venv_qaihm/bin/python scripts/download_qairt_speech_models.py \
  --qaihm-cli "$PWD/.venv_qaihm/bin/qai-hub-models"
)
```

The command downloads the pinned QCS9075 speech contexts and tokenizer files into `external_assets/models/qualcomm-ai-hub/v0.58.0`, generates their ONNX wrappers, and writes the model manifest.

In **Terminal A**, restart GenieX:

```bash
unset LD_LIBRARY_PATH ADSP_LIBRARY_PATH DSP_LIBRARY_PATH
"$HOME/.local/bin/geniex" --skip-update serve --host 127.0.0.1:18181
```

Once GenieX is ready, run the speech check and start the app in **Terminal B**:

```bash
(
set -e
unset LD_LIBRARY_PATH ADSP_LIBRARY_PATH DSP_LIBRARY_PATH
export GENIEX_TASK1_MODEL="local/qwen3vl-4b-qairt-w8a16"
export DISHCOVERY_MODELS_DIR="$PWD/external_assets/models"
export DISHCOVERY_SPEECH_MODEL_ROOT="$PWD/external_assets/models/qualcomm-ai-hub/v0.58.0"
export GENIEX_QNN_BACKEND="$HOME/.local/share/geniex/qairt/htp-files/libQnnHtp.so"
.venv_ort_qnn_245/bin/python scripts/smoke_test_qairt_speech.py \
  --model-root "$DISHCOVERY_SPEECH_MODEL_ROOT"
.venv_ort_qnn_245/bin/python scripts/start_demo.py
)
```

Open **http://127.0.0.1:8787** in the EVK browser. The speech check must pass before starting the app. Use the gallery, upload an image, or use the camera. Supported voice commands include `find ingredients`, `describe the dish`, `estimate calories`, and `execute both`.

Press Ctrl+C in each terminal to stop the app and GenieX.

## License

Copyright © 2026 Andrea Spagnolo, Danilo Pau, and STMicroelectronics S.r.l.

Original repository material is licensed under [CC BY-NC-SA 4.0](LICENSE.md). Third-party software, models, datasets, images, and trademarks retain their respective licenses.
