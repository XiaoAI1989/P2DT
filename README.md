#  Project Title
P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer


# P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer

This is the official implementation of paper "[P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer](https://arxiv.org/abs/2401.11666)". Our approach progressive prompt Decision Transformer (P2DT) aims to mitigate forgetting problem existed in the continual learning of Decision Transformer. 

### Dataset Preparation

D4RL is an open-source benchmark for offline reinforcement learning. It provides standardized environments and datasets for training and benchmarking algorithms, and it is wildlyused in offline RL methods. In this homework, we will use the D4RL dataset of the three environments to train our agent.
For each environment, D4RL provides three types of datasets, represented by expert, medium, and medium-replay. The expert dataset is collected by first training a policy using SAC, then collecting data by the full-trained agent. While the medium dataset is collected by a partially trained SAC agent, reaching around half the performance of the expert dataset. The medium-replay dataset is collected by the same partially trained SAC agent but with a different replay buffer. The medium-replay dataset consists of all samples in the replay buffer observed during training the SAC agent until reaching the “medium” performance. In this project, we only provide results on the medium dataset. 

### Replicate our results

If you want to replicate our results in our paper, you can directly run the command line for that table. For example, if you want to replicate the results of ZSCL, then run `python main.py`.


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
