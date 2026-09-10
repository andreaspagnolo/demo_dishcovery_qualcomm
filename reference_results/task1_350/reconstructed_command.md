# Task 1 W8A16 reference provenance

The JSON and CSV are the existing completed 350-image W8A16 evaluation from
`demo_dishcovery/reports/task1_evk_w8a16_350/`, imported unchanged on
2026-09-10. They are measured results, not a newly executed clean-room run of
this repository. Absolute paths in the raw report identify the source setup.

The versioned image list, ground-truth row map, and ingredient vocabulary match
that source run byte for byte. The saved settings match the current fixed
Top-20 preset. The source used a 900-second request timeout; this repository
has no client request timeout. No timeout fallback occurred in the source run.

- GenieX model: `local/qwen3vl-4b-qairt-w8a16`
- Language: INT8 weight encodings with retained INT16 exceptions; INT16 activations
- KV-cache paths: INT8; Qwen visual encoder: W8A16
- Recall: split FP16 SigLIP2 QNN contexts, 384 x 384 input
- 350 images, manifest order, seed 7; `legacy_siglip_qwen`, fixed Top-20
- 512 x 512 white letterbox, `present_possible`, 96 generated tokens
- Global row-z `threshold_ratio`: threshold 3.5, ratio 0.85, maximum 7 labels
- Visual/Qwen weights 0.25/0.75; VLM skip gap 0.25
- Micro-F1 0.7061855670103093; precision 0.7639405204460966;
  recall 0.6565495207667732; TP/FP/FN 411/127/215

Run `scripts/run_350_benchmarks.py task1` using the README environment, then
`scripts/verify_metrics.py task1 run_outputs/task1_350/eval_first_350_samples.json`.
The historical W4A16 reference is preserved in `../task1_350_w4a16/`.
