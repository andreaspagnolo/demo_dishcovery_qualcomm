# Task 2 command provenance

This is the exact command policy captured by the original `run.log` and implemented by `scripts/reproduce.py task2`: 350 ordered Food-500 rows, split FP16 SigLIP at 384, `class_caption`, SigLIP top-5, Q8_0+FP16 projector MTMD reranker, hybrid/8 threads/context 1024/batch 512/512 visual tokens/flash off, five individual calls, and `siglip_guarded` gap 3.0.

See `run.log` for the original stream and `eval_first_350_caption_alignment.json` for the complete trace.
