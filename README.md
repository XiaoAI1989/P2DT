# P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer

This is the official implementation of the ICASSP 2024 paper
"[P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer](https://arxiv.org/abs/2401.11666)".

P2DT mitigates catastrophic forgetting in the continual (task-incremental) learning of
Decision Transformers. Trajectories pass through shared **General Attention Blocks (GAB)**
that capture task-invariant knowledge, followed by **Expert Attention Blocks (EAB)** that
consume learned, task-specific **prompt tokens** appended to the input sequence. As new
tasks arrive, new prompt tokens are added while the shared backbone is protected by a
small learning rate and an EWC penalty weighted by the diagonal Fisher information.

## Repository structure

| File | Purpose |
| --- | --- |
| `model.py` | `DecisionTransformer`, GAB (`Block`) and EAB (`expert_Block`) definitions |
| `buffer.py` | trajectory replay buffer over the D4RL datasets |
| `core.py` | training loop, EWC/Fisher estimation, continual (forgetting) evaluation |
| `main.py` | entry point: sequential training over the task list + final forgetting test |
| `eval.py` | re-run the forgetting test on a finished run |
| `eval_score.py` | convert raw returns to D4RL normalized scores |
| `scripts/download_data.py` | download and convert the D4RL gym datasets |
| `tests/` | unit tests of the prompt-attention mechanism |
| `cfgs/` | hydra configs (see below) |

## Installation

```bash
conda create -n p2dt python=3.10 -y
conda activate p2dt
pip install -r requirements.txt
pip install "gymnasium[mujoco]==0.27.1"
```

If you encounter `XML Error: global coordinates no longer supported ...`, roll MuJoCo back:

```bash
pip install mujoco==2.3.3
```

## Dataset preparation

We use the [D4RL](https://github.com/Farama-Foundation/D4RL) gym locomotion datasets
(HalfCheetah, Hopper, Walker2d). The paper reports results on the `medium` datasets
(collected by a half-trained SAC policy). To download and convert them into the pickle
format expected by `buffer.py` (no mujoco-py / d4rl install required):

```bash
pip install datasets   # one-off, only needed for the download script
python scripts/download_data.py --dataset medium
```

This writes `dataset/halfcheetah-medium.pkl`, `dataset/hopper-medium.pkl` and
`dataset/walker2d-medium.pkl`.

## Training and evaluation

```bash
python main.py
```

trains the default task sequence from the paper:

1. `HalfCheetah`
2. `Hopper`
3. `Walker2d`

The run also finishes with the forgetting test (every task re-evaluated with the final
shared backbone). Other task orders from the paper can be run with hydra overrides, e.g.

```bash
python main.py "train.env_name=[Walker2d,Hopper,HalfCheetah]" "rtg_target=[5000,3600,12000]"
```

Experiments are managed with [hydra](https://hydra.cc): each run is stored under
`runs/<date>/<time>_/`, including the resolved config, `main.log`, the training-curve
plot `results.png` and the `models/` checkpoints. To re-run only the forgetting test of
a finished run:

```bash
python eval.py                      # latest run
python eval.py --run-dir runs/...   # specific run
```

## Configurations

| Config | Meaning |
| --- | --- |
| `cfgs/config.yaml` | full P2DT (2 GAB + 3 EAB, prompt length 20, EWC) |
| `cfgs/origin.yaml` | sequential Decision Transformer baseline (no prompts, no EWC) |
| `cfgs/no_prompt.yaml` | ablation: identical pipeline with `prompt_len: 0`, isolating the prompt contribution |

Select a config with `python main.py --config-name <name>` and edit hyperparameters in the yaml files.

## Implementation notes

* **Prompt-visible attention mask.** Task prompt tokens are appended at the end of the
  EAB input sequence (paper Sec. 2.3). Since the prompts are learned parameters rather
  than future timesteps, the causal mask is extended so that every trajectory position
  may attend to the prompt positions; under a plain causal mask, suffix prompts would
  be invisible to the trajectory and the mechanism would degenerate to a no-op.
  Causality between trajectory timesteps is unchanged (verified in `tests/`).
* **Per-task modules.** The three environments have different state/action
  dimensionalities, so the input embeddings and prediction heads are necessarily
  task-specific (`core.TASK_SPECIFIC_KEYWORDS`) and are stored with each task's
  checkpoint; the transformer blocks are shared across tasks. The prompt tokens
  themselves add 7,680 parameters per task (3 EABs x 20 tokens x 128 dims).
* **Progressive prompt initialization** (`train.progressive_prompt_init`): the prompt of
  task *i* is warm-started from the learned prompt of task *i-1*, realizing forward
  transfer between consecutive tasks. `DecisionTransformer.add_task_token()` grows the
  prompt pool dynamically when tasks beyond the initial list arrive.
* **EWC.** The diagonal Fisher information is estimated after each task finishes
  (`train.fisher_batches` batches at the task's final weights), matching the paper's
  definition, and the penalty is applied to the shared parameters only.
## Tests

```bash
python tests/test_mask_numpy.py   # mask-logic checks, numpy only
python tests/test_model.py        # end-to-end model checks, requires torch
```

## Citation

```bibtex
@article{wang2024p2dt,
  title={P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer},
  author={Wang, Zhiyuan and Qu, Xiaoyang and Xiao, Jing and Chen, Bokui and Wang, Jianzong},
  journal={arXiv preprint arXiv:2401.11666},
  year={2024}
}
```

## Acknowledgement

Accepted by the 49th IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2024).
