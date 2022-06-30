import os
import random
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from data_gen import create_data
from gnn import GNN
from utils import calc_rates

# set the parameters
random_seed = 1357531 # random seed
m = 6 # number of transmitters
n = m # number of receivers (equal to number of transmitters in this paper)
T = 100 # number of time slots for each configuration
density_mode = 'var_density' # density mode (either 'var_density' or 'fixed_density')
num_samples = {'train': 256, 'test': 128} # number of train/test samples
BW = 10e6 # bandwidth (Hz)
N = -174 - 30 + 10 * np.log10(BW) # Noise PSD = -174 dBm/Hz
noise_var = np.power(10, N / 10) # noise variance
P_max = np.power(10, (10 - 30) / 10) # maximum transmit power = 10 dBm
batch_size = 128 # batch size
num_features_list = [1] + [64] * 2 # number of GNN features in different layers
num_epochs = 100 # number of training epochs
f_min = .75 # minimum-rate constraint
lr_main = 1e-1 / m # learning rate for primal model parameters
lr_dual = 2 # learning rate for dual variables
T_0 = 5 # size of the iteration window for averaging recent rates for dual variable updates

# set network area side length based on the density mode
if density_mode == 'var_density':
    R = 500
elif density_mode == 'fixed_density':
    R = 1000 * np.sqrt(m / 20)
else:
    raise Exception

# set the random seed
os.environ['PYTHONHASHSEED']=str(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# create folders to save the data, results, and final model
os.makedirs('./data', exist_ok=True)
os.makedirs('./results', exist_ok=True)
os.makedirs('./models', exist_ok=True)

# create a string indicating the main experiment (hyper)parameters
experiment_name = 'm_{}_T_{}_fmin_{}_train_{}_test_{}_mode_{}'.format(m,
                                                                      T,
                                                                      f_min,
                                                                      num_samples['train'],
                                                                      num_samples['test'],
                                                                      density_mode
                                                                     )

# create PyTorch Geometric datasets and dataloaders
print('Generating the training and evaluation data ...')
path = './data/{}.json'.format(experiment_name)
loader, baseline_rates = create_data(m, n, T, R, path, num_samples, batch_size, P_max, noise_var)

# set the computation device and create the model using a GNN parameterization
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GNN(num_features_list, P_max).to(device)

# start training and evaluation
all_epoch_results = defaultdict(list)
print('Starting the training and evaluation process ...')
for epoch in tqdm(range(num_epochs)):
    for phase in loader:
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        all_variables = defaultdict(list)
        for data, batch_idx in loader[phase]:
            
            model.zero_grad()
            data = data.to(device)
            y, edge_index_l, edge_weight_l, edge_index, \
            edge_weight, a, a_l, transmitters_index, num_graphs = \
                data.y, data.edge_index_l, data.edge_weight_l, data.edge_index, data.edge_weight, \
                data.weighted_adjacency, data.weighted_adjacency_l, \
                data.transmitters_index, data.num_graphs
            
            # track history only during the training phase
            with torch.set_grad_enabled(phase == 'train'):
            
                all_rates = []
                avg_rates = []
                min_rates = []
                all_Ps = []
                
                # initialize the dual variables
                if phase == 'train':
                    mu = torch.rand(num_graphs * m, 1).to(device)
                else:
                    mu = torch.zeros(num_graphs * m, 1).to(device)
                    
                test_mu_over_time = []
                for t in range(T):
                    # pass the instantaneous fading arrays (network states), augmented with dual variables, at each step into the main GNN to get RRM decisions
                    p = model(mu.detach(), edge_index[t], edge_weight[t], transmitters_index)
                    gamma = torch.ones(num_graphs * n, 1).to(device) # user selection decisions are always the same (single Rx per Tx)
                    # calculate the resulting rates
                    rates = calc_rates(p, gamma, a[:, :, :, t], noise_var)

                    # save the rates and RRM decisions
                    all_rates.append(rates)
                    all_Ps.append(p)
            
                    if phase != 'train':
                        # update the dual variables
                        if (t + 1) % T_0 == 0:
                            avg_rates_recent = torch.mean(torch.stack(all_rates[-T_0:], dim=0), dim=0)
                            mu -= lr_dual * (avg_rates_recent.detach() - f_min)
                        # ensure non-negativity of the dual variables
                        mu.data.clamp_(0)
                        test_mu_over_time.append(mu.detach().cpu())
                
                if phase != 'train':
                    test_mu_over_time = torch.stack(test_mu_over_time, dim=0)
                all_rates = torch.stack(all_rates, dim=0)
                all_Ps = torch.stack(all_Ps, dim=0)
                avg_rates = torch.mean(all_rates, dim=0) # ergodic average rates
                
                if phase == 'train':
                    # calculate the Lagrangian
                    U = torch.sum(avg_rates.view(-1, n), dim=1) # sum-rate utility
                    f_min_constraint_term = torch.sum((mu * (avg_rates - f_min)).view(-1, n), dim=1)
                    L = torch.mean(U + f_min_constraint_term) # Lagrangian
                    L.backward() # calculate the gradients                    
                    # perform gradient ascent on model parameters
                    with torch.no_grad():
                        for i, theta_main in enumerate(list(model.parameters())):
                            if theta_main.grad is not None:
                                theta_main += lr_main * theta_main.grad

                    # zero the gradients after updating
                    for theta_ in list(model.parameters()):
                        if theta_.grad is not None:
                            theta_.grad.zero_()
                            
            # save the results within the epoch
            all_variables['mu'].append(torch.mean(mu).item())
            all_variables['rate'].extend(avg_rates.detach().cpu().numpy().tolist())
            
            if phase != 'train':
                all_variables['test_mu_over_time'].append(test_mu_over_time.squeeze(-1).T.detach().cpu().numpy())
                all_variables['all_rates'].append(all_rates.squeeze(-1).T.detach().cpu().numpy())
                all_variables['all_Ps'].append(all_Ps.squeeze(-1).T.detach().cpu().numpy())
                
        # save across-epoch results
        for key in all_variables:
            if key == 'rate':
                all_epoch_results[phase, 'rate_mean'].append(np.mean(all_variables['rate']))
                all_epoch_results[phase, 'rate_min'].append(np.min(all_variables['rate']))
                all_epoch_results[phase, 'rate_5th_percentile'].append(np.percentile(all_variables['rate'], 5))
            elif key in ['test_mu_over_time', 'all_rates', 'all_Ps']:
                all_epoch_results[phase, 'test_mu_over_time'] = all_variables['test_mu_over_time']
                all_epoch_results[phase, 'all_rates'] = all_variables['all_rates']
                all_epoch_results[phase, 'all_Ps'] = all_variables['all_Ps']
            else:
                all_epoch_results[phase, key].append(np.mean(all_variables[key]))

    # decay the learning rate by half every 100 epochs (if applicable)
    if (epoch + 1) % 100 == 0:
        lr_main *= 0.5

    # save the results and overwrite the saved model with the current model 
    torch.save([all_epoch_results, baseline_rates], './results/{}.json'.format(experiment_name))
    torch.save(model.state_dict(), './models/{}.pt'.format(experiment_name))

print('Training complete!')