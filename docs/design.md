# LeWM latent-STL experiment design

This is a separate experiment workspace. The existing `/data3/kaifany/LatentSTL`
and `/data3/kaifany/ImageWAM` source and evidence directories are read-only inputs
to setup, and are never experiment output locations.

The implementation follows the [LeWorldModel repository](https://github.com/linzhongzi/le-worldmodel)
at `ca231f9f9d9ab041034b6d05e90b6e04bd6cff82` and its
[paper](https://arxiv.org/abs/2603.19312). Upstream `jepa.py` and `module.py`
remain unchanged; the local adapter supplies data preprocessing and a direct
Hugging Face ViT constructor. This avoids coupling LIBERO collection to the
upstream benchmark environment and Lightning training stack.

The first model uses the upstream projected CLS representation. It is a global
latent, so the initial preservation predicate may respond to arm movement as
well as mug displacement. Actual simulator trajectory evaluation must establish
whether this predicate is useful. Support for token distances in the logic
module does not imply this global predictor produces spatial token dynamics.

## Initial choices

- One fixed camera, 128 x 128 RGB, patch size 16.
- Raw executed OSC_POSE actions; action metadata comes from the simulator.
- Three context actions and four observations per training window.
- Independent seeds 0, 1, 2, one RTX 2080 Ti per run; no DDP.
- Batch 32. Batch 64/128 are memory probes, not automatic recipe changes.
- FP16 AMP for the neural network; FP32 prediction loss and SIGReg.
- Images and actions are the only training inputs. Simulator diagnostics are
  stored separately for held-out measurement.
- Episode splits prevent adjacent windows of an episode appearing in both
  training and validation. Same-start ranking banks remain held out.
- CEM terminal distance, eventual reach, and reach-plus-preserve use the same
  bounds and search budget. Exact and smooth STL are separate configurations.

The original pipeline's spatial-distance, mug-preservation, and differentiable
endpoint findings motivated this experiment. The new model is trained locally;
a PushT/Reacher checkpoint is not treated as a LIBERO dynamics model.

## Validation order and interpretation

Environment verification covers CUDA FP16 on every GPU and actual headless
MuJoCo rendering. Training verification covers finite forward/backward updates,
checkpoint reload, and memory measurements. Neither is a scientific success.

After training, evaluate real encoded trajectories (Level 0), predicted versus
real latent trajectories and robustness at multiple horizons (Level 1), then
same-start candidate ranking (Level 2). Physical labels are diagnostic outcomes;
they must not enter the planner or set the main method's action scores.

Closed-loop MPC is conditional on useful earlier evidence. A declining loss or
uniform robustness sign cannot establish useful safety or reaching predicates.
Report missing label classes, representation collapse, poor ranking, and
unreliable horizons explicitly. Language grounding is outside this experiment.
