# LIBERO mug preservation data

This experiment uses the established living-room mug obstacle scene copied from
`LatentSTL/experiments/libero/mug_obstacle_table_move_arm.bddl` into the new
project. The existing repository and evidence directories are read-only. LIBERO
itself is installed from the isolated `third_party/LIBERO` checkout.

The Panda arm starts near `[-0.05, 0.22, 0.54]` metres and the diagnostic reach
target is `[-0.05, -0.22, 0.54]`. A red coffee mug occupies the straight path.
These coordinates are used for scripted data collection and held-out diagnostics,
not as model inputs or latent-planner scores. Initial arm positions have seeded
independent jitter of up to 0.012 metres per coordinate. The goal image is obtained
by actually executing a safe elevated path. Collection stops if that reference
does not reach while preserving the mug.

## Observation and executed action

- Single fixed `agentview`, 128 × 128 RGB uint8.
- Images use the existing LIBERO/ImageSTL 180-degree rotation convention.
- Raw normalized environment `OSC_POSE` action, seven components:
  `delta_xyz`, world-frame axis-angle rotation, gripper.
- Every action component is bounded by the actual environment's `[-1, 1]` bounds.
- The controller maps translation to ±0.05 metres and rotation to ±0.5 radians
  per control command; these are target deltas, not guaranteed realized motion.
- Control frequency and image sampling are both 20 Hz. There is no action repeat.
- No action is inferred from images or substituted after the simulator executes.

Camera pose, field of view, controller scales, frequencies, BDDL hash, seeds, and
the complete simulator configuration are stored with the dataset.

## Collection and splits

```bash
source scripts/env.sh
.venv/bin/python -m latent_stl_lewm.collect \
  --output datasets/libero_mug_v1 \
  --train-episodes 256 --val-episodes 32 --test-episodes 32 \
  --steps 96 --seed 20261004 --render-gpu 2
```

This collects 30,720 transitions. The initial smoke measured roughly 70 simulated
control steps per wall-clock second; the full collection takes about 8 minutes.
An existing output directory is refused. The HDF5 file is named
`trajectories.partial.h5` while collecting and atomically renamed to
`trajectories.h5` after completion. The training loader rejects files without the
completion attribute.

Each bank contains eight same-start candidates: no-op, low motion, safe elevated
reach, direct collision path, near-collision elevated path, lateral reach, smooth
random control, and randomized reach. Random controls include varied magnitudes,
directions, small rotations, and gripper commands. Policy names describe intended
coverage; actual safety/reaching labels always come from separately stored
diagnostics. For example, near-collision controls can be either safe or unsafe.

All trajectories in a bank have the same simulator state and identical initial
RGB. Banks are assigned to exactly one split. Scene seeds and initial snapshots
are disjoint across training, validation, and test. The validation and test banks
support offline candidate ranking without borrowing training trajectories.

## HDF5 schema

```text
attrs: complete, schema_version, action_dim, action_low, action_high,
       camera, control_freq_hz, frame_freq_hz, action_repeat, metadata_json
references/
  safe_rgb                       uint8 [128,128,3]
  goal_rgb                       uint8 [128,128,3]
  diagnostics/goal_eef_pos        float [3]
start_snapshots/<bank_id>/
  sim_state, integration_state, gripper_actions, controller_initial_joints,
  initial_object_pos, initial_object_quat, control_step, scene_seed
trajectories/<six_digit_id>/
  attrs: split, bank_id, policy, scene_seed, trajectory_seed, scene_id,
         start_state_sha256
  rgb                            uint8 [T+1,128,128,3]
  action                         float32 [T,7]
  done                           uint8 [T]
  control_step                   int32 [T+1]
  diagnostics/
    eef_pos, eef_quat, joint_pos, object_pos, object_quat,
    contact_count, robot_mug_contact_count, object_displacement,
    object_rotation_deg, goal_error, safe, reach
```

`rgb[t]`, `action[t]`, and `rgb[t+1]` form one transition. Trajectory windows never
cross episode boundaries. The PyTorch dataset returns only `rgb` and `action`;
RGB becomes float32 `[0,1]` and channel-first. Action normalization statistics use
training trajectories exclusively, with a 0.1 standard-deviation floor.

Diagnostic preservation means displacement ≤0.002 metres and angular change
≤2 degrees from the staged start. Diagnostic reaching means end-effector goal
error ≤0.03 metres. A whole-trajectory physical label uses `safe.all()` and
`reach.any()`. These labels do not establish that a learned latent predicate is
valid. Whole-image preservation may also penalize desired arm motion; the
separate Level-0 evaluation must expose that representation failure.

## Restore verification and inspection

The original flattened MuJoCo snapshot includes only time, qpos, and qvel. Testing
that alone exposed unequal replay. The new wrapper additionally saves/restores
`mjSTATE_INTEGRATION`, including controls, applied forces, and solver warm start,
then restores controller/gripper caches and resets observable sampling state.
Three repeated restores followed by the same sixteen actions produced bitwise
identical initial RGB, terminal RGB, and MuJoCo states. The real simulator test is
opt-in so normal unit tests do not need EGL:

```bash
source scripts/env.sh
RUN_SIMULATOR_TESTS=1 .venv/bin/python -m pytest tests/test_data.py -q
.venv/bin/python -m latent_stl_lewm.dataset \
  datasets/libero_mug_v1/trajectories.h5 \
  --output docs/data_audit.json --videos-dir runs/data_examples_v1
```

The audit checks alignment, raw action bounds, exact initial image equality within
banks, split isolation, and actual physical class coverage. It exports one video
for each observed safety/reach class. The initial `data/smoke_v1` collection
predated the complete-state restore fix and is for schema/training plumbing only;
it must not be used as scientific candidate-ranking evidence.
