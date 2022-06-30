# State-Augmented Learnable Algorithms for Resource Management in Wireless Networks

This repository contains the source code for learning state-augmented resource management algorithms in wireless networks via graph neural network (GNN) parameterizations. In particular, a GNN policy is trained, which takes as input both the network state at each time step (e.g., the channel gains across the network), as well as the dual variables indicating how much each user satisfies/violates its minimum-rate requirements over time. If run for a long-enough period of time and under mild assumptions, the algorithm is guaranteed to generated resource management decisions that are both feasible (i.e., satisfy the per-user minimum-rate requirements) and near-optimal (i.e., achieve network-wide performance within a constant additive gap of optimum). Please refer to [the accompanying paper](https://arxiv.org/abs/2207.XXXXX) for more details.

## Training and Evaluation

To train the RRM policies for networks with `m` transmitters and `n` receivers, run the following command:

```
python main.py
```

## Dependencies

* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
* [Scatter](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html)
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [tqdm](https://tqdm.github.io/)
* [Matplotlib](https://matplotlib.org/)

## Citation

Please use the following BibTeX citation if you use this repository in your work:

```
@article{StateAugmented_RRM_GNN_naderializadeh2022,
  title={State-Augmented Learnable Algorithms for Resource Management in Wireless Networks},
  author={Navid Naderializadeh and Mark Eisen and Alejandro Ribeiro},
  journal={arXiv preprint arXiv:2207.XXXXX},
  year={2022}
}
```
