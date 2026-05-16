<div align="center">

<h1>
  IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation
</h1>

<a href="https://github.com/ZGC-EmbodyAI/IntentVLA">
  <img alt="GitHub" src="https://img.shields.io/badge/GitHub-ZGC--EmbodyAI%2FIntentVLA-blue?logo=github">
</a>
<a href="https://arxiv.org/abs/2605.14712">
  <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.14712-b31b1b.svg">
</a>
<a href="#aliasbench">
  <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-AliasBench-orange">
</a>

**Shijie Lian**<sup>1,2,*</sup>
**Bin Yu**<sup>2,4,*</sup>
**Xiaopeng Lin**<sup>5,2,*</sup>
**Zhaolong Shen**<sup>2,6,*</sup>
**Laurence Tianruo Yang**<sup>1,7,†</sup><br>
**Yurun Jin**<sup>3,9</sup>
**Haishan Liu**<sup>2</sup>
**Changti Wu**<sup>2,8</sup>
**Hang Yuan**<sup>2,8</sup>
**Cong Huang**<sup>2,3</sup>
**Kai Chen**<sup>2,3,10,†</sup>

<sup>1</sup>HUST, <sup>2</sup>ZGCA, <sup>3</sup>ZGCI, <sup>4</sup>HIT, <sup>5</sup>HKUST(GZ), <sup>6</sup>BUAA, <sup>7</sup>ZZU, <sup>8</sup>ECNU, <sup>9</sup>USTC, <sup>10</sup>DeepCybo

<sup>*</sup>Equal contribution, <sup>†</sup>Corresponding author

</div>

---

## News

- [May 2026] We release the **AliasBench** benchmark code for evaluating short-horizon observation aliasing in robot manipulation.
- IntentVLA model training and evaluation code is **coming soon**.

## Abstract

Robot imitation data are often multimodal: similar visual-language observations may be followed by different action chunks because human demonstrators act with different short-horizon intents, task phases, or recent context. Existing frame-conditioned VLA policies infer each chunk from the current observation and instruction alone, so under partial observability they may resample different intents across adjacent replanning steps, leading to inter-chunk conflict and unstable execution.

We introduce **IntentVLA**, a history-conditioned VLA framework that encodes recent visual observations into a compact short-horizon intent representation and uses it to condition chunk generation. We further introduce **AliasBench**, a 12-task ambiguity-aware benchmark built on RoboTwin2 with matched training data and evaluation environments that isolate short-horizon observation aliasing. Across AliasBench, SimplerEnv, LIBERO, and RoboCasa, IntentVLA improves rollout stability and outperforms strong VLA baselines.

This repository currently releases the **AliasBench task code**. The full IntentVLA model implementation will be released soon.

## Core Contributions

- **Observation aliasing failure mode.** We identify a failure mode of frame-conditioned chunk policies under partial observability: demonstrations are multimodal across episodes but locally committed within each episode, while frame-only conditioning can break this commitment at test time.
- **AliasBench.** We construct a 12-task benchmark on RoboTwin2 for evaluating VLA behavior under short-horizon observation aliasing, together with matched simulation training data and evaluation environments.
- **IntentVLA.** We propose a history-conditioned imitation learning framework that learns a compact short-horizon intent representation from recent visual observations and uses it to condition chunk generation.
- **Extensive evaluation.** We evaluate on AliasBench, SimplerEnv, LIBERO, and RoboCasa, including ambiguous-intent tasks that directly test chunk-to-chunk consistency.

## AliasBench

AliasBench is designed to test whether a policy can preserve a consistent local continuation when the current observation is aliased. The benchmark contains states where two observations are visually similar, but the correct next action differs because of recent task context.

The benchmark contains four task families:

| Family | # Tasks | Core latent factor | Representative tasks |
| :--- | :---: | :--- | :--- |
| Back-and-Forth | 4 | Current local phase | Cook Bread and Plate It, Use Stapler and Return It |
| Crossing-Path | 3 | Recent source or origin | Move Block to Other Grid, Move Phone Between Stand and Pad |
| Bimanual | 2 | Handoff direction or side of origin | Hand Over Roller, Hand Over Pill Bottle |
| Multi-Goal | 3 | Transient cue or recently observed hidden property | Pick Flashed Blocks, Inspect Label and Place Block |

### Tasks

| Task file | Task name | Family |
| :--- | :--- | :--- |
| `place_block_aba_grid` | Move Block Out and Back | Back-and-Forth |
| `bread_skillet_plate_aba` | Cook Bread and Plate It | Back-and-Forth |
| `place_stapler_aba_pad` | Use Stapler and Return It | Back-and-Forth |
| `shoe_box_out_back` | Store Shoe and Take It Back | Back-and-Forth |
| `place_block_other_grid` | Move Block to Other Grid | Crossing-Path |
| `place_block_opposite_grid` | Move Block to Opposite Grid | Crossing-Path |
| `place_phone_other_stand` | Move Phone Between Stand and Pad | Crossing-Path |
| `move_pillbottle_abc_bimanual` | Hand Over Pill Bottle | Bimanual |
| `bimanual_handover_roller_lr` | Hand Over Roller | Bimanual |
| `pick_marked_blocks_to_line` | Pick Flashed Blocks | Multi-Goal |
| `pick_marked_cans_to_line` | Pick Flashed Cans | Multi-Goal |
| `place_block_by_hidden_label` | Inspect Label and Place Block | Multi-Goal |

## Performance

### AliasBench

All methods are trained or attempted under the same setting: **100 trajectories per task**, **30K training steps**, **16 NVIDIA H100 GPUs**, and **batch size 16 per GPU**.

| Method | Back-and-Forth | Crossing-Path | Bimanual | Multi-Goal | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen3VL-GR00T | 6.0 | 15.7 | 5.5 | 8.7 | 9.0 |
| + last 16 history frames | OOM | OOM | OOM | OOM | OOM |
| + last 8 history frames | OOM | OOM | OOM | OOM | OOM |
| + last 4 history frames | 7.3 | 19.3 | 2.5 | 11.0 | 10.4 |
| + 4 frames uniformly sampled from last 16 | 31.8 | 47.3 | 6.0 | 18.7 | 28.1 |
| **IntentVLA** | **49.3** | **74.7** | **17.0** | **31.3** | **45.8** |

`OOM` indicates that feeding long histories directly into the Qwen backbone as extra visual context runs out of GPU memory under the training setting above.

### Standard Benchmarks

| Benchmark | Metric | IntentVLA |
| :--- | :--- | :---: |
| SimplerEnv | Avg. success | 72.9 |
| LIBERO-Long | Avg@500 success | 97.4 |
| RoboCasa-GR1 Tabletop | Avg@50 success | 57.0 |

## Installation

AliasBench is implemented as a set of RoboTwin2 tasks. To use it, first install RoboTwin2 following the official installation guide:

- RoboTwin repository: https://github.com/RoboTwin-Platform/RoboTwin
- RoboTwin installation guide: https://robotwin-platform.github.io/doc/usage/robotwin-install.html

After RoboTwin is installed, copy the three folders in this repository into the corresponding folders of your RoboTwin checkout. The files in `task_config/` are intended to replace RoboTwin's default `demo_clean.yml` and `_eval_step_limit.yml` so that the AliasBench collection and evaluation limits are used. If you want to preserve the original RoboTwin settings, back up those two files before copying.

### Linux / macOS

```bash
# Assume the current directory is this repository.
# Replace /path/to/RoboTwin with your RoboTwin root.
cp -rf description/* /path/to/RoboTwin/description/
cp -rf envs/* /path/to/RoboTwin/envs/
cp -f task_config/demo_clean.yml /path/to/RoboTwin/task_config/demo_clean.yml
cp -f task_config/_eval_step_limit.yml /path/to/RoboTwin/task_config/_eval_step_limit.yml
```

### Windows PowerShell

```powershell
# Assume the current directory is this repository.
# Replace C:\path\to\RoboTwin with your RoboTwin root.
Copy-Item -Recurse -Force .\description\* C:\path\to\RoboTwin\description\
Copy-Item -Recurse -Force .\envs\* C:\path\to\RoboTwin\envs\
Copy-Item -Force .\task_config\demo_clean.yml C:\path\to\RoboTwin\task_config\demo_clean.yml
Copy-Item -Force .\task_config\_eval_step_limit.yml C:\path\to\RoboTwin\task_config\_eval_step_limit.yml
```

## Usage

After copying the task files into RoboTwin, use the standard RoboTwin data-collection and evaluation workflow.

For example:

```bash
cd /path/to/RoboTwin
bash collect_data.sh place_block_aba_grid demo_clean 0
```

You can replace `place_block_aba_grid` with any AliasBench task listed above.

## Repository Structure

```text
IntentVLA/
├── description/
│   └── task_instruction/        # Natural-language task instructions
├── envs/                        # AliasBench RoboTwin task definitions
├── task_config/                 # Task configs and evaluation limits
└── README.md
```

## Model Code

The IntentVLA model code, training configs, and evaluation scripts are **coming soon**.

## Acknowledgements

AliasBench is built on top of **RoboTwin2**. We thank the RoboTwin team for releasing an extensible robot manipulation benchmark and simulation platform.

## Citation

If you find this project useful, please cite our paper:

```bibtex
@misc{IntentVLA_2026_arXiv,
  title         = {IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation},
  author        = {Lian, Shijie and Yu, Bin and Lin, Xiaopeng and Shen, Zhaolong and Yang, Laurence Tianruo and Jin, Yurun and Liu, Haishan and Wu, Changti and Yuan, Hang and Huang, Cong and Chen, Kai},
  year          = {2026},
  eprint        = {2605.14712},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2605.14712}
}
```
