# State-Augmented Learnable Algorithms for Resource Management in Wireless Networks

This repository contains the source code for learning state-augmented resource management algorithms in wireless networks via graph neural network (GNN) parameterizations. In particular, a GNN policy is trained, which takes as input both the network state at each time step (e.g., the channel gains across the network), as well as the dual variables indicating how much each user satisfies/violates its minimum-rate requirements over time. If run for a long-enough period of time and under mild assumptions, the algorithm is guaranteed to generated resource management decisions that are both feasible (i.e., satisfy the per-user minimum-rate requirements) and near-optimal (i.e., achieve network-wide performance within a constant additive gap of optimum). Please refer to [the accompanying paper](https://arxiv.org/abs/2207.02242) for more details.

## Training, Evaluation, and Visualization of Results

To train the state-augmented power control algorithms and evaluate their performance on a set of test configurations, run the following command:

```
python main.py
```

The wireless network parameters (such as the number of transmitter-receiver pairs) and the learning hyperparameters (such as the numbers and sizes of GNN hidden layers) can be adjusted in the first few lines of `main.py`. Upon completion of the training and evaluation process, the generated dataset, the entire results, and the final model parameters are saved in separate folders named `data`, `results`, and `models`, respectively.

In order to visualize the results, as an example, the following code (in a Python environment) can be used to plot the results with the default set of parameters in `main.py`:

```
from plot_gen import plot_results
plot_results('./results/m_6_T_100_fmin_0.75_train_256_test_128_mode_var_density.json')
```

The result should look like the following figure:

![sample_results_figure](https://user-images.githubusercontent.com/67436161/176785223-c190d2cb-2667-4bb9-9cab-bc911a25c0f5.png)

## Dependencies

* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
* [Scatter](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html)
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [tqdm](https://tqdm.github.io/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

## Citation

Please use the following BibTeX citation to cite the accompanying paper if you use this repository in your work:

```
@article{StateAugmented_RRM_GNN_naderializadeh2022,
  title={State-Augmented Learnable Algorithms for Resource Management in Wireless Networks},
  author={Navid Naderializadeh and Mark Eisen and Alejandro Ribeiro},
  journal={arXiv preprint arXiv:2207.02242},
  year={2022}
}
```
