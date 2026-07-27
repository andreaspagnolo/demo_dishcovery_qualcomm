# EVK results on the 350-image difficulty subset

Report date: 2026-07-22

This report records the EVK results obtained on the 350-image difficulty
subset. The two tasks use the same image subset and ground-truth mapping, but
their metrics are task-specific.

## Summary

| Task | Configuration | Primary result |
| --- | --- | ---: |
| Task 1 — ingredient recognition | Legacy paper logic, fixed top-20 candidates | Micro-F1 **0.6601** |
| Task 2 — caption retrieval | `siglip_guarded`, gap 3.0, SigLIP top-5, reranker top-5 | Caption accuracy **0.6771** |

## Dataset and reproducibility

- Images: `benchmark_inputs/task1_350/images.txt`
- Image count: 350
- Evaluation order: first 350 entries, seed 7 for Task 1 and seed 42 in the
  archived Task 2 evaluation artifact
- Task 1 ground-truth map: `labels/image_ground_truth_rows.csv`
- The 350-image manifest SHA-256 is
  `ebbdde7f4b54b16c3d1d2cacef7eb06642617a4f46fffa26925a5e2ec2342f47`.

## Task 1 — ingredient recognition

### Configuration

The EVK run uses the archived paper logic, with EVK-specific SigLIP and Qwen
models:

```text
task1_logic=legacy_siglip_qwen
candidate_list_mode=fixed_topk
top_k=20
vlm_prompt_mode=present_possible
vlm_max_new_tokens=96
vlm_max_pixels=262144 (512 x 512)
visual_weight=0.25
qwen_weight=0.75
selector=threshold_ratio
selector_score_mode=row_z
selector_threshold=3.5
selector_ratio=0.85
selector_max_labels=7
final_selection_scope=global
final_rank_depth=20
skip_vlm_rel_gap_threshold=0.25
skip_vlm_visual_selector=top_ratio
skip_vlm_visual_selector_ratio=0.85
skip_vlm_visual_selector_max_labels=3
```

### Accuracy

| Metric | Value |
| --- | ---: |
| Micro-F1 | **0.660054** |
| Precision | 0.757764 |
| Recall | 0.584664 |
| Row-average F1 | 0.762242 |
| True positives | 366 |
| False positives | 117 |
| False negatives | 260 |
| Exact-match rows | 208 / 350 |
| Mean predicted labels | 1.38 |
| Mean ground-truth labels | 1.7886 |
| Top-20 recall | 0.920128 |
| Top-20 all-truth rows | 317 / 350 |

### Runtime

| Quantity | Value |
| --- | ---: |
| Mean SigLIP2 image/top-k time | 1.1998 s/image |
| Mean Qwen time | 2.4788 s/image |
| Mean total image time | 3.6837 s/image |
| Total accounted evaluation time | 1293.68 s |
| VLM-skipped rows | 171 / 350 (48.86%) |

The raw result is available at
[`eval_first_350_samples.json`](task1_350/eval_first_350_samples.json),
with predictions in
[`eval_first_350_samples_predictions.csv`](task1_350/eval_first_350_samples_predictions.csv).

The EVK models are not the Orin engines: the EVK uses QNN SigLIP2 and the
Qualcomm QAIRT W4A16 Qwen3-VL model through GenieX. Therefore this score is an
EVK result under aligned task logic, not a bit-identical reproduction of the
Orin TensorRT result.

## Task 2 — caption retrieval and reranking

The authoritative 350-image run is the Q8 reranker evaluation. It uses the
requested guarded candidate policy:

```text
caption text mode: class_caption (caption-ranking mode)
final_score_mode=siglip_guarded
siglip_keep_gap=3.0
siglip_top_k=5
rerank_top_k=5
rerank_weight=0.7
```

The EVK stack for this result is the Q8_0 GGUF Qwen3-VL-Reranker-2B with an
FP16 multimodal projector and hybrid mtmd execution. The archived Orin path
uses its patched next-token true/false logits; the EVK reranker returns its
image-caption pair score, so the metric is comparable at the task level but
not numerically identical at the model-logit level.

### Accuracy

| Metric | Value |
| --- | ---: |
| Caption top-1 accuracy | **0.677143** (237 / 350) |
| Dish-class top-1 accuracy | **0.877143** (307 / 350) |
| SigLIP top-1 caption accuracy before reranking | 0.545714 (191 / 350) |
| Truth caption in rerank candidates | 303 / 350 (0.865714) |
| Reranked pairs | 1355 |
| Guard-skipped rows | 79 |
| Failed rows | 0 |

### Runtime

| Quantity | Value |
| --- | ---: |
| Mean end-to-end latency | 7.9668 s/image |
| Median latency | 9.7384 s/image |
| P95 latency | 11.0815 s/image |
| Maximum latency | 13.3428 s/image |
| Throughput | 0.1239 images/s |
| Measured wall time | 2824.51 s |

The raw result is available at
[`eval_first_350_caption_alignment.json`](task2_350/eval_first_350_caption_alignment.json).

## Interpretation

Task 1 reaches high candidate recall (0.9201) but loses recall during the
EVK Qwen selection and fusion stage, producing micro-F1 0.6601. Task 2
reranking improves caption accuracy from the SigLIP-only 0.5457 baseline to
0.6771, while class-level accuracy reaches 0.8771.

These are EVK measurements with Qualcomm/QNN model artifacts. Differences from
Orin should be attributed to the changed quantized models, runtimes, and
fixed-shape preprocessing in addition to the common task configuration.
