# Task 1 command provenance

No original console log was retained. This command is reconstructed from the complete archived JSON settings and is implemented verbatim by `scripts/reproduce.py task1`.

- first 350 Task 1 manifest entries; seed 7
- `legacy_siglip_qwen` preset with fixed top-20 candidates
- split FP16 SigLIP QNN contexts, 384 RGB bicubic preprocessing
- GenieX alias `local/qwen3vl-4b-qairt-w4a16`; 512 white letterbox; 96 tokens
- row-z global `threshold_ratio`: threshold 3.5, ratio 0.85, max 7
- visual/Qwen weights 0.25/0.75; VLM skip gap 0.25

The archived result is `eval_first_350_samples.json`; use the launcher, not this note, to run it.
