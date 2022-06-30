import numpy as np
import torch
from numpy import linalg as LA
from torch_geometric.data import Data, Dataset

# calculating rates
def calc_rates(p, gamma, h, noise_var):
    """
    calculate rates for a batch of b networks, each with m transmitters and n recievers
    inputs:
        p: bm x 1 tensor containing transmit power levels
        gamma: bn x 1 tensor containing user scheduling decisions
        h: b x (m+n) x (m+n) weighted adjacency matrix containing instantaneous channel gains
        noise_var: scalar indicating noise variance
        training: boolean variable indicating whether the models are being trained or not; during evaluation, 
        entries of gamma are forced to be integers to satisfy hard user scheudling constraints
        
    output:
        rates: bn x 1 tensor containing user rates
    """
    b = h.shape[0]
    p = p.view(b, -1, 1)
    gamma = gamma.view(b, -1, 1)
    m = p.shape[1]
    
    combined_p_gamma = torch.bmm(p, torch.transpose(gamma, 1, 2))
    signal = torch.sum(combined_p_gamma * h[:, :m, m:], dim=1)
    interference = torch.sum(combined_p_gamma * torch.transpose(h[:, m:, :m], 1, 2), dim=1)
    
    rates = torch.log2(1 + signal / (noise_var + interference)).view(-1, 1)
    
    return rates

# baseline ITLinQ method
def ITLinQ(H_raw, Pmax, noise_var, PFs):
    H = H_raw * Pmax / noise_var
    n = np.shape(H)[0]
    prity = np.argsort(PFs)[-1:-n-1:-1]
    flags = np.zeros(n)
    M = 10 ** 2.5
    eta = 0.5
    flags[prity[0]] = 1
    for pair in prity[1:]:
        SNR = H[pair,pair]
        INRs_in = [H[TP,pair] for TP in range(n) if flags[TP]]
        INRs_out = [H[pair,UE] for UE in range(n) if flags[UE]]
        max_INR_in = max(INRs_in)
        max_INR_out = max(INRs_out)
        if max(max_INR_in,max_INR_out) <= M * (SNR ** eta):
            flags[pair] = 1
    return flags * Pmax

def convert_channels(a, snr):
    a_flattened = a[a > 0]
    a_flattened_log = np.log(snr * a_flattened)
    a_norm = LA.norm(a_flattened_log)
    a_log = np.log(snr * a)
    a_log[a == 0] = 0
    return a_log / a_norm

class Data_modTxIndex(Data):
    def __init__(self,
                 y=None,
                 edge_index_l=None,
                 edge_weight_l=None,
                 edge_index=None,
                 edge_weight=None,
                 weighted_adjacency=None,
                 weighted_adjacency_l=None,
                 transmitters_index=None,
                 init_long_term_avg_rates=None,
                 num_nodes=None,
                 m=None):
        super().__init__()
        self.y = y
        self.edge_index_l = edge_index_l
        self.edge_weight_l = edge_weight_l
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.weighted_adjacency = weighted_adjacency
        self.weighted_adjacency_l = weighted_adjacency_l
        self.transmitters_index = transmitters_index
        self.init_long_term_avg_rates = init_long_term_avg_rates
        self.num_nodes = num_nodes
        self.m = m
                
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'transmitters_index':
            return self.m
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
class WirelessDataset(Dataset):
    def __init__(self, data_list):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx], idx