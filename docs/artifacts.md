# Artifacts and cleanup

GitHub contains code, tests, documentation, and compact result summaries. Large
artifacts remain local under ignored directories. A fresh clone does not contain
trained weights, collected data, videos, virtual environments, or upstream clones.
Use [workflow.md](workflow.md) to recreate the pipeline.

## Retained local artifacts

Paths below are relative to the repository root on the research machine.

| Path | Purpose |
| --- | --- |
| `datasets/libero_mug_v1/trajectories.h5` | Original 256/32/32 train/validation/test trajectories |
| `runs/campaign_20261004T073606Z/` | Three training seeds, checkpoints, and original evaluations |
| `runs/cem_improved_20261005T064943Z/multistep_training/best.pt` | Frozen-representation multi-step dynamics checkpoint |
| `runs/cem_improved_20261005T064943Z/` | Reference banks, action priors, and CEM improvement trials |
| `runs/latent_stl_research_20261005T075000Z/` | Frozen final CEM comparison, replays, source snapshots, and audits |
| `runs/visual_stl_benchmark_20261006T071302Z/` | Completed 864-trajectory visual benchmark and forecasts |
| `runs/cem_comparison_20261004T225600Z/` | Local versus official pretrained checkpoint comparison |
| `runs/dynamics_research_20261005/` | Counterexamples, dynamics diagnostics, and failed alternatives |
| `runs/research_*`, `runs/analysis_*`, `runs/replays_*` | Development evidence referenced by the research reports |
| `.cache/pretrained/` | Downloaded official Cube checkpoint; excluded from Git |

Each main campaign has a local `index.html` or JSON reports. Historical artifacts
may record absolute machine paths; their original provenance is intentionally
preserved. They are not dependencies bundled with a GitHub checkout.

## Cleanup policy

Redundant `runs/*smoke*` outputs, the obsolete `data/smoke_v1` fixture, temporary
collection logs, stale launch notes, and copied unused source-inspection files
were removed. Small useful audit and memory-measurement summaries were retained
under [evidence/](evidence/). Automated tests remain in `tests/`.

Failed research experiments are retained: they document limitations and help
prevent reporting only successes. Cleanup does not delete the original dataset,
trained checkpoints, held-out evaluations, reference artifacts, or canonical
replays. `runs/repository_cleanup.json` records the local deletion inventory.
