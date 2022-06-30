import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
plt.rcParams.update({'font.size': 14})

def plot_results(path):
    all_epoch_results, baseline_rates = torch.load(path)
    ylabels = ['Mean rate (bps/Hz)', 'Minimum rate (bps/Hz)', '5th percentile rate (bps/Hz)']
    
    # get results for the proposed method
    res = [None for _ in range(3)]
    res[0] = all_epoch_results['test', 'rate_mean']
    res[1] = all_epoch_results['test', 'rate_min']
    res[2] = all_epoch_results['test', 'rate_5th_percentile']
    
    # plot the results side by side
    fig, ax = plt.subplots(1, 3, figsize=(25, 5))
    fig.patch.set_facecolor('white')
    for i in range(3):
        ax[i].grid(True)
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel(ylabels[i])
        ax[i].plot(res[i], label='State-Augmented')
        for alg in baseline_rates:
            r = baseline_rates[alg]
            if i == 0:
                metric = np.mean(r)
            elif i == 1:
                metric = np.min(r)
            elif i == 2:
                metric = np.percentile(r, 5)
            ax[i].plot([metric] * len(res[i]), label=alg)

    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fancybox=True)
    plt.show()

    