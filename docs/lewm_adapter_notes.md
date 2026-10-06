# LeWM integration and verification

The adapter imports `JEPA` from `third_party/le-worldmodel/jepa.py` and
`ARPredictor`, `Embedder`, `MLP`, and `SIGReg` from its `module.py`. These upstream
files are unchanged. It constructs the same Hugging Face ViT-tiny backbone
directly, avoiding the upstream Lightning/data-framework dependency stack.

The backbone factory was checked against the primary source
[`stable_pretraining/backbone/utils.py`](https://github.com/galilai-group/stable-pretraining/blob/ab836bf699a2be712dfcf980a5eb70d36391e876/stable_pretraining/backbone/utils.py):
192 hidden units, 12 layers, 3 attention heads, 768 intermediate units, default
Hugging Face initialization, and no pooler. The source is linked above rather than
vendored. The default adapter contains **18,032,144 parameters**; this is the
measured count of this configuration, not the paper's approximate headline count.

Deliberate adaptations for LIBERO/RTX 2080 Ti:

- One 128-by-128 RGB camera; patch size 16 exactly tiles the image. The upstream
  default is 224-by-224 with patch size 14.
- FP16 autocast and GradScaler for forward/backward; MSE, embedding statistics,
  and SIGReg characteristic-function trigonometry are explicitly FP32.
- Three context frames and their aligned actions predict the next three latents
  from four observed frames. The target receives gradients, matching upstream.
- Training-only action mean/std, with std floored at 0.1 for constant or nearly
  constant action dimensions. Raw environment actions remain the public API;
  normalization is internal and checkpointed. ImageNet normalization is internal.
- Whole episodes use the collector's fixed train/val/test split. Dataset readers
  expose only RGB and action tensors. Physics diagnostics never enter the model.
- AdamW at 5e-5, weight decay 1e-3, gradient clipping at 1.0, step-based linear
  warmup/cosine schedule. This small independent trainer replaces the upstream
  epoch-based Lightning orchestration and does not use DDP.

`encode` returns a projected global CLS vector, as in the official LeWM objective.
It does not claim to predict spatial tokens. `rollout(z0, actions)` includes the
initial latent and returns `B,H+1,D`. Each predicted successor uses the most recent
three latent/action pairs. Image inputs are `B,C,H,W` or `B,T,C,H,W`, uint8 or float
in [0,1]. Use `model.eval()` for inference. Diagnostic state inputs are rejected.

The `frozen_encoder_learned_predictor` backend freezes the encoder **and projector**,
including BatchNorm running statistics. The training CLI requires a checkpoint
with prior optimizer updates, rather than labeling randomly frozen weights as
pretrained. The initial implementation supports a locally trained LeWM encoder;
external V-JEPA encoder integration is a later backend.

Checkpoints contain configuration, model/optimizer/scaler states, RNG states,
deterministic sampler position, source hashes, dataset hash, upstream revision,
and run metadata. `load_checkpoint(path, device)` returns `(model, checkpoint)`;
`load_model` returns the model. `best.pt` minimizes held-out total prediction plus
SIGReg loss; `last.pt` supports resuming. Collapse diagnostics (`latent_std` and
SIGReg) are logged separately. Validation defaults to all held-out windows.

Verification on October 4, 2026:

- Eight behavioral tests passed: temporal/action alignment, official loss equality,
  finite encoder gradients, learned action dependence, frozen BatchNorm state,
  input scaling, state-dict roundtrip, and sampler resume behavior.
- Full-architecture FP16 forward/backward/AdamW benchmarks passed at batches
  32, 64, and 128 with peak allocated memory of 1.085, 1.915, and 3.568 GiB
  respectively. Peak reserved memory was 1.227, 2.127, and 3.936 GiB.
- Forty real-image training updates were finite with no skipped AMP updates.
  A checkpoint after update 30 was reloaded with optimizer/scaler/RNG/sampler
  states and continued through update 40. The final checkpoint also loaded on CPU.

The obsolete plumbing fixtures and smoke checkpoints have been removed.
Production data is `datasets/libero_mug_v1`. Historical memory measurements are
preserved in [evidence/memory_benchmark.json](evidence/memory_benchmark.json).
