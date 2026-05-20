# RESA_mmWave — Master Reference

**Last updated:** 2026-05-19  
**Repository:** `/home/hullumdr/resa/RESA_mmWave`  
**Status:** first-pass refactor of the prior `LLM_ML/jetson_nav_pipeline/canon` / `3branch` radar navigation stack.

This repository reorganizes the previous mmWave wheelchair-navigation code by responsibility:

- `branch1/` — radar point generation, camera/radar autolabeling, Branch 1 point classifier, and Branch 1 training utilities
- `branch2/` — live navigation loop, scene aggregation, map memory, directness, and motion adapters
- `branch3/` — dense Range-Doppler free-space U-Net and verbal guidance/TTS helpers
- `sim_eval/` — replay, visualization, and evaluator scripts
- `config/` — radar/camera calibration and radar profile assets

The code is source-complete enough to describe the intended architecture, but it is not yet a clean runnable package. Several modules still import the old `models.*`, `perception.*`, `scene_pipeline`, and `guidance` names without the package aliases/directories that this refactor requires.

---

## Table of Contents

1. [System Goal](#1-system-goal)
2. [Current Repository Layout](#2-current-repository-layout)
3. [End-to-End Architecture](#3-end-to-end-architecture)
4. [Branch 1 — Point Cloud Classifier and Data Pipeline](#4-branch-1--point-cloud-classifier-and-data-pipeline)
5. [Branch 2 — Live Loop, Scene Pipeline, and MapBuilder](#5-branch-2--live-loop-scene-pipeline-and-mapbuilder)
6. [Branch 3 — Dense RD Free-Space and Guidance](#6-branch-3--dense-rd-free-space-and-guidance)
7. [Simulation and Evaluation](#7-simulation-and-evaluation)
8. [Configuration and Hardware Contracts](#8-configuration-and-hardware-contracts)
9. [Known Refactor Gaps](#9-known-refactor-gaps)
10. [Essential Missing Artifacts or Code](#10-essential-missing-artifacts-or-code)
11. [Recommended Next Refactor Steps](#11-recommended-next-refactor-steps)
12. [File Reference](#12-file-reference)

---

## 1. System Goal

The intended system is a radar-first navigation stack for a wheelchair-mounted TI IWR1843 mmWave radar. At deployment time the system is radar-only. RGB/depth camera data is used for calibration, offline synchronization, segmentation-based autolabeling, and training-data construction.

The high-level runtime loop is:

```text
radar_only_recorder
  -> adc_to_pointcloud_v6 / RD sidecar export
  -> Branch 1 point classifier
  -> Branch 2 MapBuilder + scene aggregation + deterministic nav decision
  -> Branch 3 U-Net advisory free-space signal
  -> guidance / TTS
```

Coordinate convention inherited from the prior repo:

- `x` = lateral, right-positive
- `y` = forward range, away from radar
- `z` = up, positive above radar
- below-floor multipath cutoff in live loop: `MIN_VALID_Z = -0.851`
- live actionable range gate in `branch2/navigation_loop.py`: `0.3 m <= range <= 5.0 m`

---

## 2. Current Repository Layout

```text
RESA_mmWave/
├── branch1/
│   ├── calib/                 # camera/radar calibration and spatiotemporal calibration
│   ├── inputs/
│   │   ├── geometry/          # soft structural and depth correspondence helpers
│   │   └── recorders/         # radar-only and DCA + RealSense recorders
│   ├── models/
│   │   ├── 3branch_common.py  # consolidated KPConv temporal model + dataset utilities
│   │   ├── 3dgcnn/            # legacy DGCNN/KPConv common modules
│   │   ├── kpconv/            # pure-PyTorch KPConv frame encoders
│   │   └── hybrid rd/         # RD/RA patch model, runtime, and dataset code
│   ├── processing/            # ADC processing, tensor export, autolabel prep, OneFormer scripts
│   ├── tools/                 # dataset, four-class, directness/depth, spatiotemporal helpers
│   └── training/              # Branch 1 training driver
├── branch2/
│   ├── navigation_loop.py     # intended live loop entry point
│   ├── scene_pipeline.py      # logger, aggregation, nav decision, EgoVelocityEstimator, MapBuilder
│   ├── directness_runtime.py  # ego/directness feature logic
│   └── scene_motion_adapter.py
├── branch3/
│   ├── branch3_unet.py        # BEV U-Net + checkpoint helpers + sector probabilities
│   └── guidance.py            # LLM guidance wrapper, fast fallback, Piper/espeak TTS
├── sim_eval/
│   ├── branch3_replay_simulation.py
│   ├── evaluate_replay_decisions_against_oneformer.py
│   └── visualize_seg_radar_overlay.py
└── config/
    ├── profile_objdet.cfg
    ├── radar_camera_extrinsics.json
    ├── camera_intrinsics_d435i.json
    └── hard_negative_reviewer.html
```

There is no repo-level `README`, dependency file, package metadata, or installed package namespace yet.

---

## 3. End-to-End Architecture

The previous repo described three branches. This refactor preserves that conceptual split, but physically separates the implementation:

| Branch | Current location | Role |
|---|---|---|
| Branch 1 | `branch1/` | raw ADC to point cloud, training/autolabeling, temporal KPConv point classification |
| Branch 2 | `branch2/` | live orchestration, scene aggregation, deterministic decision, persistent map memory |
| Branch 3 | `branch3/` | dense RD free-space U-Net and guidance/TTS helpers |
| Simulation | `sim_eval/` | replay videos, OneFormer comparison, radar/video overlay |

The intended runtime control path is `branch2/navigation_loop.py`. That file expects:

- radar session data from `branch1/inputs/recorders/radar_only_recorder.py`
- point-cloud generation from `branch1/processing/adc_to_pointcloud_v6.py`
- Branch 1 model code from a `models` package
- Branch 1 runtime RD/RA helper code from a `models.hybrid_rd_runtime` module
- soft-structural and calibration helpers from a `perception` package
- `scene_pipeline.py` and `guidance.py` importable by bare module names
- optional Branch 3 checkpoint and helper functions

Those paths are not currently wired into a coherent import namespace.

---

## 4. Branch 1 — Point Cloud Classifier and Data Pipeline

### Branch 1 purpose

Branch 1 classifies sparse radar detections into semantic point classes. The refactor contains two overlapping Branch 1 contracts:

1. `branch1/models/3branch_common.py` documents and implements an older 3-class contract:
   - `structure`
   - `floor`
   - `human`
   - 18 scalar features
   - optional RD and RA patch encoders for 82 total augmented features

2. `branch1/training/train_hybrid_rd_patch_kpconv.py` trains the newer directness/RD-scalar contract:
   - supports `base`, `soft_structural`, and `directness_soft_structural`
   - `directness_soft_structural` = 29 scalar features
   - supports 4-class `bucket_4class` style training if the dataset/checkpoint supplies it
   - imports old package names that are not valid in this repo as-is

This means the refactor currently preserves both historical and current Branch 1 code paths, but has not reconciled them into one canonical model package.

### Core files

| File | Role |
|---|---|
| `branch1/processing/adc_to_pointcloud_v6.py` | ADC binary/session processor; exposes `RadarProcessor`, `build_processor`, and `process_session_fast` |
| `branch1/processing/export_radar_tensors.py` | tensor export path |
| `branch1/processing/export_radar_tensors_data511.py` | batch sidecar export helper; still assumes old `canon`/`models` layout |
| `branch1/processing/prepare_autolabel.py` | builds synchronized metadata for autolabeling; still has old default path assumptions |
| `branch1/processing/ghost_teacher_fusion.py` | ghost/directness label fusion helper |
| `branch1/processing/noah_scripts/OneFormer/*` | OneFormer segmentation export, autolabeling, and QC scripts |
| `branch1/models/3branch_common.py` | consolidated temporal KPConv model, datasets, feature utilities, patch encoders |
| `branch1/models/kpconv/3branch_frame_encoder.py` | pure-PyTorch KPConv frame encoder |
| `branch1/models/3dgcnn/*` | legacy DGCNN/KPConv common code |
| `branch1/models/hybrid rd/*` | hybrid RD/RA patch model wrapper, runtime extraction, patch dataset code |
| `branch1/training/train_hybrid_rd_patch_kpconv.py` | active-looking Branch 1 training driver, but import paths are not refactor-clean |
| `branch1/tools/dataset_tools.py` | dataset and patch-index construction helpers |
| `branch1/tools/branch1_fourclass_tools.py` | 4-class label/dataset utilities |
| `branch1/tools/depth_correspondence_tools.py` | depth correspondence utility wrapper |
| `branch1/tools/spatiotemporal_tools.py` | spatiotemporal helper CLI |
| `branch1/calib/*` | static and spatiotemporal calibration |

### Data and label path

The intended training-data flow remains:

```text
record DCA1000 + RealSense session
  -> export OneFormer segmentation label maps
  -> adc_to_pointcloud_v6.py generates per-session radar CSV and RD sidecars
  -> prepare_autolabel.py synchronizes radar/video/depth metadata
  -> noah_autolabel_radar_using_synccsv_v4_depth_gated.py projects radar points into labels
  -> dataset_tools.py builds point datasets and RD/RA patch indices
  -> train_hybrid_rd_patch_kpconv.py trains temporal KPConv checkpoint
```

The scripts still point at historical data roots such as `LLM_ML/data/...`; this repo does not contain datasets.

---

## 5. Branch 2 — Live Loop, Scene Pipeline, and MapBuilder

### Branch 2 purpose

Branch 2 is the integration layer. It:

- starts/stops live radar recording
- processes each session into point clouds
- loads the Branch 1 checkpoint
- computes runtime features and optional RD/RA patches
- updates persistent map memory
- aggregates classified points into a scene dict
- computes deterministic navigation decisions
- queues guidance and TTS

### Core files

| File | Role |
|---|---|
| `branch2/navigation_loop.py` | intended live Jetson runtime entry point |
| `branch2/scene_pipeline.py` | logger, scene aggregator, decision policy, `OccupancyGrid`, `EgoVelocityEstimator`, `MapBuilder` |
| `branch2/directness_runtime.py` | ego-motion/directness feature estimation |
| `branch2/scene_motion_adapter.py` | adapters for scene-level motion metadata |

### Decision policy

`branch2/scene_pipeline.py` contains `compute_nav_decision(scene)`. Its deterministic priority is broadly inherited from the previous repo:

1. floor not visible -> stop
2. center map novel obstacle -> stop
3. close center human -> stop
4. center blocked and map confirms -> stop
5. center blocked and U-Net disagrees -> veer with lower urgency
6. center blocked -> veer to open side, or stop if both sides blocked
7. approaching human -> slow down
8. novel free space -> slow down
9. low confidence -> slow down
10. otherwise continue

The LLM/guidance layer should verbalize this decision, not decide the action.

### MapBuilder

`MapBuilder` in `branch2/scene_pipeline.py` is a `threading.Thread` with queue-based updates. It maintains structure/free evidence grids, estimates ego velocity, supports per-point occupancy scores, and exposes anomaly queries. It is the refactored Branch 2 memory component.

Important inherited caveat: this implementation appears to use post-update grid state for anomaly queries. If the project needs strict "novel before this frame" semantics, it should add a pre-update snapshot before accumulation.

---

## 6. Branch 3 — Dense RD Free-Space and Guidance

### Branch 3 purpose

Branch 3 predicts dense free-space from RD tensors and supplies an advisory signal to the decision stack. It is not the primary blocker detector; Branch 1 and Branch 2 remain primary.

### Core files

| File | Role |
|---|---|
| `branch3/branch3_unet.py` | U-Net with spatial attention; RD cube input helpers; checkpoint loader; sector summarization |
| `branch3/guidance.py` | LLM guidance text, fast rule-based fallback, Piper/espeak TTS |

### U-Net contract

`branch3/branch3_unet.py` implements:

- `BEVUNet(in_channels=24, base_ch=32, out_h=128, out_w=128)`
- input RD tensor shape: `(B, 24, 32, 256)` for real/imag of 12 virtual channels
- optional prior support through `in_channels=25`
- output free-space probability map: `(B, 128, 128)`
- `sector_probabilities()` converts the map to left/center/right/free summary values

No trained U-Net checkpoint is present in this repo.

### Guidance contract

`branch3/guidance.py` contains:

- `build_scene_description(payload)`
- `GuidanceEngine` using Ollama at `http://127.0.0.1:11435`
- `FastGuidanceEngine` fallback
- `TTSEngine` using Piper first, then espeak fallbacks

The module imports `scene_pipeline` by bare name, so it currently expects `branch2/` to be on `PYTHONPATH` or the file to be moved/aliased.

---

## 7. Simulation and Evaluation

`sim_eval/` is intended to hold non-live validation tools:

| File | Role |
|---|---|
| `sim_eval/branch3_replay_simulation.py` | replay recorded sessions through the integrated pipeline and write MP4 artifacts |
| `sim_eval/evaluate_replay_decisions_against_oneformer.py` | compare replay decisions to OneFormer-derived evidence |
| `sim_eval/visualize_seg_radar_overlay.py` | draw radar points over video/segmentation |

The replay script still assumes a local `sim_eval/models/`, `sim_eval/config/`, and `sim_eval/perception/` layout because `THIS_DIR` is `sim_eval/`. In this refactor those assets live under sibling directories (`branch1/`, `branch2/`, `branch3/`, `config/`), so replay is not expected to run without path fixes.

---

## 8. Configuration and Hardware Contracts

Config files currently preserved:

| File | Role |
|---|---|
| `config/profile_objdet.cfg` | TI radar profile used by ADC processing and live capture |
| `config/radar_camera_extrinsics.json` | radar-to-camera extrinsic calibration |
| `config/camera_intrinsics_d435i.json` | RealSense D435i camera intrinsics |
| `config/hard_negative_reviewer.html` | browser UI for reviewing hard negatives |

Hard-coded deployment paths still present:

- `/home/ryan/nav_sessions/`
- `/home/ryan/xwr/profile_objdet.cfg`
- `/home/ryan/xwr/ml_model`
- `/home/ryan/xwr/llm`
- `/home/ryan/miniconda3/envs/mmwave/bin/piper`
- `/home/ryan/piper_voices/en_US-lessac-medium.onnx`

Those paths should become environment variables or repo-local defaults before this fork can be used outside the original Jetson layout.

---

## 9. Known Refactor Gaps

These are code-level gaps observed in this repository, not just documentation differences.

### Import namespace mismatch

Many files still use the old package names:

- `from models.hybrid_rd_model import ...`
- `from models.hybrid_rd_runtime import ...`
- `from models.radar_3dgcnn_common_kpconv import ...`
- `from perception.adc_to_pointcloud_v6 import ...`
- `from perception.corridor_soft_structural import ...`
- `from scene_pipeline import ...`
- `from guidance import ...`

The refactor has no repo-level `models/` package and no repo-level `perception/` package. The code was moved into branch-specific folders but imports were not consistently updated.

### `branch1/models/hybrid rd/` cannot be imported as a Python package

The directory name contains a space. Python package imports such as `models.hybrid_rd_model` cannot resolve this location. Rename it to `hybrid_rd/` or create a package shim.

### Missing package initialization and entry-point strategy

There are no `__init__.py` files and no `pyproject.toml`. Some scripts use ad hoc `sys.path.insert`, but those inserts point to historical layouts rather than the current fork.

### Branch 1 label/feature contract conflict

`branch1/models/3branch_common.py` documents 18 scalar features and a 3-class label order. The training driver supports 29-feature directness/RD scalar training and newer 4-class datasets. Decide whether the canonical Branch 1 contract is:

- legacy 3-class / 18-feature / 82 augmented, or
- current 4-class / 29-feature / RD+RA augmented.

The previous guides identify the 4-class directness/RD/true-RA path as the intended current path.

### Live loop still references old Jetson layout

`branch2/navigation_loop.py` points at `/home/ryan/...` and expects models under `branch2/models/`, which does not exist. It also imports `perception.*` and `models.*` modules that are actually under `branch1/`.

### Replay script path roots are wrong for this refactor

`sim_eval/branch3_replay_simulation.py` sets `PIPELINE_DIR = THIS_DIR`, so it searches inside `sim_eval/` for `models`, `config`, and `perception`. Those directories are not there.

### Branch 3 file naming mismatch

The U-Net file is `branch3/branch3_unet.py`, while older code often expected `models/branch3_unet.py` or `3branch_unet.py` under a pipeline folder.

### No dependency manifest

The code uses at least `numpy`, `pandas`, `torch`, `sklearn`, `scipy`, `cv2`, and `tqdm`, plus hardware/runtime tools such as Piper, espeak, Ollama, TI DCA1000 capture tooling, and OneFormer/transformers assets. There is no requirements or environment file.

---

## 10. Essential Missing Artifacts or Code

The following are essential for end-to-end use and are not present in this repository:

1. Trained Branch 1 checkpoint, commonly expected as `3branch_best_model.pt` or supplied via `NAV_POINTCLOUD_RD_PATCH_MODEL_PT`.
2. Trained Branch 3 U-Net checkpoint, commonly expected as `unet_best_model.pt` or supplied via `NAV_UNET_PT`.
3. A runnable package layout or compatibility shims for `models.*`, `perception.*`, `scene_pipeline`, and `guidance`.
4. A renamed/importable hybrid RD module directory; `branch1/models/hybrid rd/` should not remain as a package target.
5. Deployment dependency manifest and environment setup.
6. Dataset artifacts for training/replay, including recorded session directories, radar CSVs, RD/RA sidecars, segmentation outputs, and labeled point CSVs.
7. OneFormer large local model assets if `oneformer_seg_large.py` is intended to run offline; only a partial `oneformer_large/merges.txt` appears to be present.
8. Hardware capture dependencies and config deployment scripts for the Jetson/DCA1000 environment.

The source code may also be missing the prior repo's fully wired `canon/` entry-point glue. If this fork is intended to replace `canon/`, the refactor needs a thin compatibility package or explicit imports from `branch1`, `branch2`, and `branch3`.

---

## 11. Recommended Next Refactor Steps

1. Choose a canonical package layout. A practical target:

   ```text
   resa_mmwave/
   ├── branch1/
   ├── branch2/
   ├── branch3/
   ├── sim_eval/
   └── config/
   ```

   Then make imports explicit, for example `from resa_mmwave.branch1.processing.adc_to_pointcloud_v6 import ...`.

2. Rename `branch1/models/hybrid rd/` to `branch1/models/hybrid_rd/` and update imports.

3. Add `__init__.py` files and a minimal `pyproject.toml` so scripts can be run with `python -m ...`.

4. Move hard-coded `/home/ryan/...` deployment paths behind environment variables with repo-local defaults.

5. Reconcile Branch 1 around one active contract. Based on the prior master guide, the likely target is 4-class directness/soft-structural/RD-scalar plus RD/true-RA patch augmentation.

6. Put compatibility checks in each entry point:

   - verify radar config exists
   - verify model checkpoint exists
   - verify expected package imports resolve
   - verify sidecar paths are present for RD/RA checkpoints

7. Add a small smoke test suite:

   - import all top-level modules
   - instantiate `BEVUNet`
   - instantiate `MapBuilder`
   - parse `config/profile_objdet.cfg`
   - run `adc_to_pointcloud_v6.py --help`
   - run `train_hybrid_rd_patch_kpconv.py --help`

8. Decide whether `sim_eval/` should call the live code directly or use a dedicated replay facade. Right now it is halfway between both.

---

## 12. File Reference

### Active-looking files

| Path | Notes |
|---|---|
| `branch2/navigation_loop.py` | intended live loop, but imports/path constants need refactor |
| `branch2/scene_pipeline.py` | most self-contained Branch 2 file; includes logger, aggregator, decision, MapBuilder |
| `branch3/branch3_unet.py` | self-contained model architecture and helpers |
| `branch3/guidance.py` | guidance/TTS, needs `scene_pipeline` import path |
| `branch1/processing/adc_to_pointcloud_v6.py` | likely core radar DSP entry point |
| `branch1/training/train_hybrid_rd_patch_kpconv.py` | intended current Branch 1 trainer, needs import path fixes |
| `branch1/tools/dataset_tools.py` | dataset construction utility, needs import path fixes |
| `sim_eval/branch3_replay_simulation.py` | useful replay harness, but path roots still assume old layout |

### Legacy or transitional files

| Path | Notes |
|---|---|
| `branch1/models/3dgcnn/*` | legacy model paths kept for checkpoint compatibility |
| `branch1/models/3branch_common.py` | consolidated but appears behind the newer training contract |
| `branch1/processing/noah_scripts/*` | many historical autolabel variants preserved |
| `branch1/processing/export_radar_tensors_data511.py` | batch helper with old `canon` path assumptions |
| `config/hard_negative_reviewer.html` | standalone UI asset, references old script names in comments/text |

