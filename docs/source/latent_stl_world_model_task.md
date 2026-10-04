# Task: Latent Signal Temporal Logic with a Learned World Model

## Objective

Build a first proof-of-concept for **Latent Signal Temporal Logic (Latent STL)** in the existing ImageSTL / ImageWAM LIBERO research codebase.

The central hypothesis is:

> A robot task can be specified with temporal logic over learned visual/world-model latents instead of hand-written geometric state predicates.

The desired pipeline is:

```text
current observation
    |
    v
world-model encoder
    |
    v
z_0
    |
candidate action sequence U
    |
    v
action-conditioned latent world model
    |
    v
predicted latent trajectory z_hat_0:H
    |
    v
latent predicates
    |
    v
STL robustness rho_phi(z_hat_0:H)
    |
    v
planner / optimizer
    |
    v
execute first action(s), observe again, replan
```

The first implementation should **not** attempt open-vocabulary language grounding or a giant foundation world model. The goal is to establish whether temporal logic can be evaluated on a predicted latent trajectory and used for planning.

---

# 1. Research question

Traditional STL writes predicates over known physical state, for example:

```math
mu_reach(x_t) = r_g - ||x_t - x_goal||_2
```

```math
mu_safe(x_t) = ||x_t - x_obs||_2 - r_obs
```

and composes them:

```math
phi = Always_[0,H](mu_safe) AND Eventually_[0,H](mu_reach)
```

This requires manually exposing and engineering quantities such as:

- end-effector pose,
- object pose,
- obstacle geometry,
- object orientation,
- task thresholds,
- collision geometry.

This project instead evaluates predicates in latent space.

For an observation encoder `E`:

```math
z_t = E(o_t)
```

define reference-image predicates such as:

```math
mu_reach(z_t) = delta_goal - d(z_t, z_goal)
```

```math
mu_avoid(z_t) = d(z_t, z_violation) - delta_violation
```

```math
mu_preserve(z_t) = delta_preserve - d(z_t, z_start)
```

and evaluate:

```math
phi = Always_[0,H](mu_preserve) AND Eventually_[0,H](mu_reach)
```

The important new ingredient is that the latent sequence should be **predicted from actions**:

```math
z_hat_{t+1} = F(z_hat_t, a_t, optional_state_t)
```

rather than obtained by fully executing every candidate in MuJoCo.

---

# 2. Existing project context to preserve

Before writing code, inspect the existing repository and understand the current image-goal planning pipeline.

Relevant files are expected to include or resemble:

```text
experiments/libero/svgd_endpoint.py
experiments/libero/svgd_endpoint_differentiable.py
experiments/libero/svgd_obstacle_path.py
experiments/libero/svgd_terminal_q.py
experiments/libero/score_endpoint_candidates.py
experiments/libero/sample_endpoint_trajectories.py
experiments/libero/scene_geometry.py
experiments/libero/latent_jacobian.py
```

Also inspect the current experiment documentation and run artifacts before changing interfaces.

Important existing findings to preserve conceptually:

1. Goal-image latent distance can rank and optimize simple endpoint-reaching trajectories.
2. Spatial/token-wise latent distance has generally been more useful than a global scalar image distance.
3. A latent reference representing an undisturbed scene has already acted as an obstacle/preservation constraint.
4. Long open-loop trajectory optimization has been difficult when temporal credit is not differentiable.
5. Physical state should remain a held-out diagnostic when evaluating whether an image/latent-only objective works.
6. Existing evidence directories and canonical runs must never be overwritten.

Do not alter old run artifacts. Use new versioned output directories.

---

# 3. Recommended world model for the first experiment

## Primary choice: LeWorldModel (LeWM)

Use **LeWorldModel** as the first implementation target.

Why:

- It is a JEPA-style action-conditioned world model.
- It predicts future states in latent space.
- It is small, roughly ~15M parameters.
- It is designed for latent-space MPC / CEM planning.
- Its official implementation includes environment integration, data loading, planning, and pretrained checkpoints for several benchmark environments.
- It is small enough to train on one consumer GPU, making it appropriate for the current hardware.

Repository / project to investigate:

```text
https://github.com/linzhongzi/le-worldmodel
```

Paper:

```text
LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels
arXiv:2603.19312
```

Do not assume that a released PushT / Reacher / Cube checkpoint transfers directly to LIBERO.

For the LIBERO experiment, the action-conditioned dynamics must match the local:

- camera distribution,
- robot embodiment,
- action definition,
- controller,
- scene dynamics,
- temporal sampling rate.

Therefore the likely first useful model is:

```text
LeWM architecture
+ locally collected LIBERO trajectories
+ local training
```

rather than a frozen off-the-shelf LeWM checkpoint.

---

# 4. Alternative world-model configurations

Implement a clean adapter interface so the experiment is not tied permanently to LeWM.

Create an interface conceptually like:

```python
class LatentWorldModel:
    def encode(self, obs):
        # RGB observation -> latent state.
        ...

    def predict_next(self, z, action, state=None):
        # Predict next latent.
        ...

    def rollout(self, z0, actions, states=None):
        # Return predicted latent trajectory z_0:H.
        ...

    def latent_distance(self, z_a, z_b, metric="cosine"):
        # Return latent distance.
        ...
```

At minimum support:

```text
backend = lewm
backend = frozen_encoder_learned_predictor
```

Later backends may include:

```text
V-JEPA 2 / V-JEPA 2.1 encoder + local predictor
V-JEPA 2-AC
DINO-based world model
```

Do not build all of these now.

---

# 5. Important clarification: what "frozen JEPA" can and cannot do

A frozen image/video JEPA encoder by itself is **not an action-conditioned world model**.

A frozen encoder provides:

```math
o_t -> z_t
```

but planning requires:

```math
(z_t, a_t) -> z_{t+1}
```

Therefore one of the following is required:

### Option A — LeWM

Train encoder and predictor together on local action-conditioned data.

### Option B — frozen visual JEPA + learned dynamics predictor

Freeze a pretrained V-JEPA / V-JEPA 2.1 encoder:

```math
z_t = E_frozen(o_t)
```

and train only:

```math
F_theta(z_t, a_t, s_t) -> z_{t+1}
```

This is a very attractive second experiment.

### Option C — frozen V-JEPA 2-AC

Use the released V-JEPA 2-AC encoder and predictor without training.

This can be attempted as a baseline, but it should not be assumed to work in LIBERO. The released model was post-trained on DROID / Franka interaction data with specific:

- image preprocessing,
- camera distribution,
- end-effector state convention,
- 7-D action convention,
- temporal sampling.

Any mismatch can invalidate planning.

For a first proof-of-concept, prefer A.

For a representation-transfer experiment, prefer B.

Treat C as an optional zero-shot baseline.

---

# 6. Scope of the first MVP

Do **not** start with language.

Do **not** start with open-vocabulary predicates.

Do **not** start with full long-horizon LIBERO manipulation.

First prove:

```text
Can latent STL evaluated on a learned predicted latent trajectory
select better actions than terminal latent distance alone?
```

Use an already understood scene.

Preferred first task:

## Mug preservation + reaching

Reuse the existing mug obstacle / living-room-style scene if it is stable.

Task:

```text
move the arm to a target region
while keeping the mug / protected object undisturbed
```

Specification:

```math
phi =
Always_[0,H](mu_preserve)
AND
Eventually_[0,H](mu_reach)
```

where:

```math
mu_preserve(z_t)
    = delta_preserve - d(z_t, z_safe_reference)
```

and:

```math
mu_reach(z_t)
    = delta_goal - d(z_t, z_goal)
```

`z_goal` and `z_safe_reference` must come only from images/latents available to the optimizer.

Physical measurements are diagnostics only.

---

# 7. Critical experiment decomposition

Do not jump directly from world model to closed-loop success.

Implement the following four levels.

## Level 0 — Ground-truth simulator latent STL

For a real simulator rollout:

```text
actions
 -> MuJoCo rollout
 -> RGB frames
 -> encoder
 -> true latent sequence
 -> latent STL robustness
```

This establishes whether the latent predicates themselves correctly score trajectories.

No world-model prediction is involved yet.

Questions:

- Does latent robustness distinguish safe from unsafe trajectories?
- Does latent robustness rank reaching trajectories above failures?
- Does `Always(preserve) AND Eventually(reach)` behave as expected?
- Does the combined temporal score outperform terminal-only distance?

This is the first mandatory validation.

---

## Level 1 — World-model prediction accuracy

Train or load the world model.

Given real trajectory prefixes and actions:

```text
z_t + a_t:t+k
 -> predicted z_hat_t+1:t+k
```

compare predicted latents to encoded ground-truth future observations.

Evaluate:

- one-step latent error,
- 2-step error,
- 4-step error,
- 8-step error,
- optionally 16-step error,
- cosine distance,
- normalized L2,
- any model-native distance.

Also compare:

```text
rho_phi(predicted latent trajectory)
vs
rho_phi(real encoded latent trajectory)
```

This robustness agreement is more important than raw prediction error.

Report:

- Pearson correlation,
- Spearman correlation,
- sign agreement for satisfied / violated trajectories,
- false-safe rate,
- false-unsafe rate.

---

## Level 2 — Offline action ranking

For the same start state, sample `K` candidate action sequences.

For every candidate:

1. Predict latent rollout with the world model.
2. Compute predicted latent-STL robustness.
3. Separately execute the candidate in MuJoCo.
4. Encode the real trajectory.
5. Compute true latent-STL robustness.
6. Record physical diagnostics.

Then ask:

```text
Does predicted robustness rank candidates similarly to actual robustness?
```

Metrics:

- Spearman rank correlation,
- top-1 regret,
- top-k recall,
- safe trajectory selection rate,
- goal success rate,
- physical constraint violations.

Do not yet run online MPC until this stage is credible.

---

## Level 3 — Closed-loop latent-STL MPC

Use receding-horizon control:

```text
observe current frame
 -> encode z_t
 -> optimize short candidate action sequences
 -> choose best sequence
 -> execute first action / short action chunk
 -> observe again
 -> repeat
```

Do not execute a long predicted trajectory open-loop.

Default policy:

```text
planning horizon: short
execute horizon: 1-4 environment steps
replan after execution
```

The horizon should be determined empirically from the Level-1 robustness-agreement curves.

---

# 8. Planner choice

## First planner: CEM

Use Cross-Entropy Method first.

Reason:

- LeWM already supports CEM-style latent planning.
- It does not require gradients through the entire predictor.
- It isolates the Latent-STL idea from optimizer-specific failure modes.
- It is simpler to compare against terminal latent-distance planning.

The candidate cost should be:

```math
J(U) = -rho_phi(z_hat_0:H(U))
```

Use bounded action sequences.

Implement:

```text
planner = cem_terminal_goal
planner = cem_latent_stl
```

with identical:

- population size,
- iterations,
- action bounds,
- horizon,
- random seed.

After this works, optionally add:

```text
planner = svgd_latent_stl
```

using autograd through the world-model predictor.

Do not make SVGD a prerequisite for the MVP.

---

# 9. STL robustness implementation

Create a small standalone latent-STL module.

Suggested path:

```text
experiments/libero/latent_stl.py
```

It should not depend on MuJoCo.

Core API:

```python
predicate_reach(...)
predicate_preserve(...)
predicate_avoid(...)

always(robustness, start, end)
eventually(robustness, start, end)
conjunction(*robustness_values)
disjunction(*robustness_values)

evaluate_formula(...)
```

Support exact robustness first:

```math
rho_Always = min_t rho_t
rho_Eventually = max_t rho_t
rho_AND = min(rho_1, rho_2)
rho_OR = max(rho_1, rho_2)
```

Also support differentiable smooth versions:

```text
smooth_min
smooth_max
temperature tau
```

Do not silently mix exact and smooth semantics.

Persist which version is used in every experiment config.

---

# 10. Latent predicates

Implement these first.

## Reach

```math
rho_reach(t) =
delta_goal - d(z_t, z_goal)
```

Positive means the latent is inside the goal tolerance.

## Preserve

Two possible forms.

### Safe-reference form

```math
rho_preserve(t) =
delta_preserve - d(z_t, z_safe)
```

### Start-preservation form

```math
rho_preserve(t) =
delta_preserve - d(z_t, z_start)
```

The safe-reference form is preferable if the arm itself moves substantially and whole-image start distance would incorrectly punish desired motion.

## Violation reference

Optional:

```math
rho_avoid(t) =
d(z_t, z_bad) - delta_bad
```

where `z_bad` is a latent encoding of a known undesirable state.

---

# 11. Spatial features

If the world model exposes spatial tokens, preserve token structure.

Prefer:

```text
[token_1, token_2, ..., token_N]
```

over collapsing everything into one vector.

Test:

```text
global cosine
global L2
mean token cosine
```

If region masks are available, support:

```math
d(M * z_t, M * z_reference)
```

for preservation predicates.

However, do not block the initial LeWM MVP if its easiest representation is global.

The first objective is to prove temporal composition, not perfect spatial localization.

---

# 12. World-model data collection in LIBERO

Create a dedicated dataset generator.

Suggested path:

```text
experiments/libero/collect_world_model_data.py
```

Required fields per trajectory:

```text
rgb[t]
action[t]
robot_state[t]          # optional model input, always useful diagnostic
done[t]
scene/task id
camera metadata
sim seed
trajectory seed
timestamp/control step
```

Also store physical state diagnostics separately:

```text
eef pose
joint positions
object poses
collision/contact information
```

These diagnostics must not become latent-STL optimizer inputs unless explicitly running a privileged baseline.

Use HDF5 or another chunked format compatible with training.

Never store only successful trajectories.

The world model needs:

- no-op / low-motion data,
- regular motion,
- safe trajectories,
- near-collision trajectories,
- constraint violations,
- different directions,
- different action magnitudes.

The data distribution must cover actions the planner may later propose.

---

# 13. Action representation

Do not hard-code action dimensionality before inspecting the existing LIBERO traces and controller.

Record and document:

```text
raw env action
controller interpretation
action bounds
control frequency
frame sampling frequency
```

For the first world-model experiment, use the same action representation the simulator actually executes.

If the current policy/controller uses a high-dimensional action that makes learning difficult, add a clearly separate experiment using a reduced task-space action, e.g.:

```text
delta_xyz
or
delta_xyz + delta_rotation + gripper
```

Do not conflate those experiments.

---

# 14. Image representation / resolution

Do not train the first LeWM experiment on the full `[agentview | wrist] 224 x 448` composite unless memory and speed are first measured.

Recommended first configuration:

```text
one fixed camera
128 x 128 or similarly small square input
```

Candidate views:

```text
agentview
birdview
sideview
```

Prefer a view where the protected object and goal-relevant motion are both visible.

After the method works, add multiview.

Possible multiview strategies:

1. independent encoder per view with shared weights,
2. encode each view then concatenate latents,
3. multi-view token set.

Avoid increasing token count unnecessarily in the first run.

---

# 15. Baselines

Every main result should compare at least:

## Baseline A — terminal latent distance

```math
J_terminal(U) = d(z_hat_H, z_goal)
```

## Baseline B — temporal latent reach only

```math
phi = Eventually(mu_reach)
```

## Method — latent STL

```math
phi = Always(mu_preserve) AND Eventually(mu_reach)
```

## Diagnostic upper bound — ground-truth encoded trajectory

Run the same latent-STL robustness on actual MuJoCo rollout frames.

This isolates:

```text
predicate failure
vs
world-model prediction failure
vs
planner failure
```

Optional privileged diagnostic:

```text
traditional physical STL
```

computed from simulator state.

Never use privileged physical STL to choose the method's actions.

---

# 16. Logging requirements

Every planner iteration / candidate evaluation should be reproducible.

Record:

```text
run config
git commit / dirty state
world model checkpoint hash
encoder checkpoint hash
goal image hash
safe reference image hash
scene seed
planner seed
action bounds
horizon
CEM population
CEM iterations
latent metric
STL formula
STL thresholds
smooth/exact semantics
```

For each selected plan record:

```text
predicted latent trajectory score
predicted predicate traces
predicted STL robustness
real encoded predicate traces
real encoded STL robustness
physical goal error
object displacement
object rotation
collision diagnostics
actual action sequence
selected action prefix
```

Save videos for selected trajectories.

---

# 17. Visualization

Create plots for:

## Predicate traces

```text
time
|
|  reach robustness
|  preserve robustness
|
+---------------- time
```

Mark zero as the satisfaction boundary.

## Predicted vs actual robustness

Scatter:

```text
x = predicted rho_phi
y = real rho_phi
```

## Planning progression

Per MPC cycle:

```text
best predicted rho
actual rho after execution
goal progress
constraint diagnostics
```

If possible, integrate with the existing trajectory viewer instead of building a completely separate UI.

---

# 18. GPU plan for current hardware

Current hardware assumption:

```text
3 x NVIDIA RTX 2080 Ti
11 GB VRAM each
```

## LeWorldModel

Expected to be the practical choice.

The published LeWM model is roughly ~15M parameters and is described as trainable on one GPU in a few hours.

For the current hardware:

```text
GPU 0: LeWM training seed 0
GPU 1: LeWM training seed 1 / ablation
GPU 2: simulator data generation, evaluation, or seed 2
```

Do not start with DDP.

Use independent runs.

Important RTX 2080 Ti note:

```text
use FP16 AMP if stable
do not assume BF16 support
```

Start with:

```text
batch size 32
```

then increase to:

```text
64
128
```

only after measuring VRAM.

If the official config defaults to 128 and OOMs, reduce the batch without changing the scientific question.

Record effective batch size.

Because the first experiment uses reduced image resolution and a small model, 11 GB should be sufficient for LeWM-scale training.

---

# 19. GPU plan for frozen V-JEPA variants

## Frozen V-JEPA / V-JEPA 2.1 encoder + small local predictor

This is a good second-stage experiment.

Recommended approach:

```text
frozen V-JEPA 2.1 ViT-B or another smaller checkpoint
+
10M-50M action-conditioned predictor
```

Advantages:

- encoder need not be trained,
- stronger pretrained visual representation,
- local predictor can learn LIBERO action dynamics,
- feasible on 11 GB if features are cached or encoder inference is run in small batches.

Strongly consider offline feature caching:

```text
RGB dataset
 -> frozen encoder once
 -> latent dataset
 -> train predictor only
```

This can make predictor training very cheap.

---

# 20. V-JEPA 2-AC baseline

The released V-JEPA 2-AC is substantially larger:

```text
ViT-g encoder ~1B parameters
action-conditioned predictor ~300M parameters
```

It is useful as a scientific baseline but should not be the first implementation target.

Possible experiment:

```text
load pretrained V-JEPA 2-AC frozen
match image preprocessing
match state/action convention
run very small CEM candidate batches
evaluate zero-shot ranking in one LIBERO Franka task
```

Do not assume zero-shot transfer.

Its released planning setup was trained around Franka/DROID conventions.

If attempted on 11 GB GPUs:

- use FP16 inference,
- use batch size 1 for encoder calls,
- encode current and goal images once,
- chunk candidate action rollouts,
- reduce CEM population initially,
- monitor peak VRAM,
- distribute independent candidate batches across GPUs if useful.

Do not attempt full V-JEPA 2-AC training on the 3 x 2080 Ti system as part of this task.

---

# 21. Language extension — phase after the latent-STL MVP

Do not require the world model itself to understand language.

Separate:

```text
dynamics
grounding
logic
```

## Dynamics

```text
world model:
(z_t, a_t) -> z_t+1
```

## Grounding

A predicate scorer:

```math
g_psi(z_t, c) -> scalar robustness
```

where `c` may be:

```text
"mug upright"
"gripper holding red mug"
"plate undisturbed"
"mug inside bowl"
```

## Logic

An LLM compiles a language instruction into a formula over a known predicate vocabulary.

Example input:

```text
"Pick up the red mug, keep it upright, avoid touching the plate,
and place it in the bowl."
```

Example compiler output:

```text
predicates:
  grasp_mug
  mug_upright
  plate_undisturbed
  mug_in_bowl

formula:
  Eventually(grasp_mug)
  AND Always(mug_upright)
  AND Always(plate_undisturbed)
  AND Eventually(mug_in_bowl)
```

Then convert to an internal AST.

Do not let the LLM directly emit arbitrary executable Python.

---

# 22. Language implementation stages

## Stage L1 — language to predefined STL

Use a fixed predicate library.

The LLM only chooses:

- predicates,
- operators,
- time windows.

This is the safest and easiest version.

## Stage L2 — text-conditioned learned predicates

Train:

```math
P_psi(z_t, text_embedding) -> robustness
```

Use contrastive / ranking supervision rather than only binary labels.

Desired ordering:

```text
very satisfied
>
barely satisfied
>
0
>
barely violated
>
strongly violated
```

## Stage L3 — open-vocabulary latent STL

Allow new language concepts to instantiate learned predicates without explicit per-task coding.

This is future work and should not be part of the first coding milestone.

---

# 23. Suggested file structure

Add something like:

```text
experiments/libero/latent_stl.py
experiments/libero/world_models/base.py
experiments/libero/world_models/lewm_adapter.py
experiments/libero/collect_world_model_data.py
experiments/libero/train_lewm_libero.py
experiments/libero/eval_world_model_rollout.py
experiments/libero/eval_latent_stl_ranking.py
experiments/libero/plan_latent_stl_cem.py
experiments/libero/test_latent_stl.py
experiments/libero/test_world_model_adapter.py
scripts/latent_stl/
    collect_data.sh
    train_lewm.sh
    eval_prediction.sh
    eval_ranking.sh
    run_mpc.sh
```

Do not modify old optimizer behavior unless required.

Prefer wrappers and shared utilities.

---

# 24. Unit tests

Minimum tests:

## STL semantics

Construct synthetic robustness arrays and verify:

```text
Always = min
Eventually = max
AND = min
OR = max
```

Test window indexing.

Test smooth versions approach exact semantics as temperature decreases.

## Predicate sign

Verify:

```text
goal distance < threshold => positive
goal distance > threshold => negative
```

Same for preserve / violation predicates.

## World-model adapter

Test shapes:

```text
encode(B, C, H, W)
predict_next(B, ...)
rollout(B, T, ...)
```

Ensure gradients are preserved when requested.

## Planner

Use a toy differentiable latent system where the optimal action is analytically obvious.

Verify CEM selects the correct direction.

---

# 25. First experimental sequence

Run in this order.

## Experiment 0 — latent STL with true simulator frames

No world model.

Goal:

```text
validate formula + predicates
```

## Experiment 1 — LeWM one-step prediction

Goal:

```text
show the local dynamics model learned meaningful latent transitions
```

## Experiment 2 — LeWM multi-step robustness prediction

Goal:

```text
measure planning horizon where predicted STL robustness remains trustworthy
```

## Experiment 3 — offline candidate ranking

Goal:

```text
predicted latent STL selects real safe/reaching trajectories
```

## Experiment 4 — closed-loop CEM latent-STL MPC

Goal:

```text
solve reach + preserve task without using privileged physical state
```

Only after these pass:

## Experiment 5 — frozen pretrained encoder + local predictor

Compare LeWM representation against a frozen V-JEPA-family encoder.

Only after that:

## Experiment 6 — language compiler

Compile language into the already validated predicate library.

---

# 26. Scientific acceptance criteria

Do not define success only as "training loss goes down."

For the first paper-quality claim, require evidence across:

### Prediction

World model predictions preserve useful task ordering over a nontrivial horizon.

### Logic

Latent STL robustness correctly distinguishes:

```text
reach + safe
reach + violation
safe + no reach
neither
```

### Ranking

Predicted robustness ranks real candidate trajectories meaningfully.

### Planning

Closed-loop latent-STL planning improves constraint satisfaction over terminal latent-distance planning at matched compute.

### No leakage

Physical coordinates remain diagnostics only.

### Generalization

Use multiple simulator / planner seeds.

---

# 27. Failure diagnosis matrix

When the final planner fails, identify the stage.

## Case A

```text
true encoded latent STL is wrong
```

=> predicate / representation failure.

## Case B

```text
true latent STL works
predicted latent STL is wrong
```

=> world-model prediction failure.

## Case C

```text
predicted latent STL ranks candidates correctly
planner still fails
```

=> optimizer / search-budget failure.

## Case D

```text
planner selects good predicted trajectory
first actions work
long rollout drifts
```

=> horizon / compounding-error failure; shorten MPC horizon and replan.

Always report which layer failed.

---

# 28. Constraints for the coding agent

1. Do not overwrite existing run directories.
2. Do not modify canonical evidence artifacts.
3. Do not use physical goal state as an optimization input in the main method.
4. Do not silently introduce simulator object pose into latent predicates.
5. Keep privileged state-based STL only as a diagnostic baseline.
6. Make every model/planner seed explicit.
7. Persist full configs and checkpoint hashes.
8. Keep language out of the initial MVP.
9. Prefer simple CEM before adding SVGD.
10. Prefer short-horizon MPC before long open-loop planning.
11. Use the smallest viable image resolution and model first.
12. Do not claim that a frozen passive JEPA encoder is an action-conditioned world model.

---

# 29. Deliverables

The coding task is complete when the repository contains:

1. A reusable latent-STL evaluator.
2. A world-model adapter abstraction.
3. A working LeWM-backed adapter or local LeWM training path.
4. A LIBERO world-model dataset generator.
5. One-step and multi-step latent prediction evaluation.
6. Predicted-vs-real STL robustness evaluation.
7. Offline candidate-ranking benchmark.
8. Closed-loop CEM latent-STL MPC runner.
9. Matched terminal-distance baseline.
10. Unit tests.
11. Reproducible launch scripts.
12. A Markdown report summarizing:
    - architecture,
    - training data,
    - GPU usage,
    - prediction horizon,
    - robustness correlation,
    - ranking results,
    - closed-loop planning results,
    - failures and next steps.

---

# 30. Final target architecture

The long-term architecture is:

```text
              natural-language task
                       |
                       v
               language compiler
                       |
                       v
        temporal logic formula phi
                       |
                       | concepts / predicates
                       v
RGB/state ---> encoder ---> z_t
                       |
                 candidate actions
                       |
                       v
           latent dynamics predictor
                       |
                       v
             z_hat_t:t+H
                       |
                       v
           semantic predicate heads
                       |
                       v
              STL robustness
                       |
                       v
           CEM / SVGD / MPC planner
                       |
                       v
              execute + replan
```

Conceptually:

```text
Language says WHAT must happen.
The latent world model predicts WHAT WOULD happen.
STL checks WHEN and WHETHER the requirements hold.
The planner chooses HOW to act.
```

The first implementation should validate the lower half of this stack before attempting the language layer.
