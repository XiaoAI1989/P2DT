#  Project Title
P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer


# P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer

This is the official implementation of paper "[P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer](https://arxiv.org/abs/2401.11666)". Our approach progressive prompt Decision Transformer (P2DT) aims to mitigate forgetting problem existed in the continual learning of Decision Transformer. 

### Dataset Preparation

D4RL is an open-source benchmark for offline reinforcement learning. It provides standardized environments and datasets for training and benchmarking algorithms, and it is wildlyused in offline RL methods. In this homework, we will use the D4RL dataset of the three environments to train our agent.
For each environment, D4RL provides three types of datasets, represented by expert, medium, and medium-replay. The expert dataset is collected by first training a policy using SAC, then collecting data by the full-trained agent. While the medium dataset is collected by a partially trained SAC agent, reaching around half the performance of the expert dataset. The medium-replay dataset is collected by the same partially trained SAC agent but with a different replay buffer. The medium-replay dataset consists of all samples in the replay buffer observed during training the SAC agent until reaching the “medium” performance. In this project, we only provide results on the medium dataset. 


# Environment

```bash
conda activate <hw4-env-name>
pip install gymnasium[mujoco]==0.27.1
```
Rollback the MuJoCo package to version 2.3.3 if you encounter the following error:
"XML Error: global coordinates no longer supported. To convert existing models, load and save them in MuJoCo 2.3.3 or older"
```bash
pip install mujoco==2.3.3
```

if you are using zsh as your shell, use the following command instead:

```zsh
conda activate <hw4-env-name>
pip install gymnasium\[mujoco\]==0.27.1
```

A successful installation of the MuJoCo package could be troublesome, but hopefully, the situation has improved a lot since the shift from the old `mujoco_py` packages. If you encounter problems installing the package, first refer to their GitHub [repo](https://github.com/deepmind/mujoco) for guidance.

### Replicate our results

If you want to replicate our results in our paper, you can directly run the command line for that table. For example, if you want to replicate the results of ZSCL, then run `python main.py`.

### About the code
we are using the hydra framework to manage the configuration of the experiments. Please refer to hydra’s documentation for more details. The results and saved files will be stored in the runs folder, under the subfolder specified by the time of execution. You can find the training curves and a video of the trained agent in the subfolder. If you want to turn off this behavior and save everything in the current folder, you can change the hydra.run.chdir field in the config.yaml file to false.

models.py This file contains the model definitions for the DecisionTransformer class. 
buffer.py This file contains the replay buffer classes we use to store D4RL trajectories and sample mini-batches from them.
utils.py This file implements a handful of tool functions we use in the training process. 
core.py This file contains the main training and evaluation loop. 
main.py This is the main file that you’ll run to start the training process. 

### Training and Evaluation

run `python main.py`

### Hyperparameters

You can modify the Hyperparameters at the yaml file in the config fold.


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
