import numpy as np
from scipy.spatial import distance
import numpy.matlib

# Global parameters
min_D_TxTx = 35 # minimum Tx-Tx distance
min_D_TxRx = 10 # minimum Tx-Rx distance
max_D_TxRx = 50 # maximum Tx-Rx distance
shadowing = 7 # shadowing standard deviation
f_c = 2.4e9 # carrier frequency (Hz)
speed = 1 # receiver speed (m/s)

def UDN_PL(D, alpha1=2):
    m, n = np.shape(D)
    L = np.zeros((m, n))
    k0 = 39
    a1 = alpha1
    a2 = 4
    db = 100

    CONST = 10 * np.log10(db ** (a2-a1))
    
    for i in range(m):
        for j in range(n):
            d = D[i,j] 
            if d <= db:
                L[i,j] = k0 + 10 * a1 * np.log10(d)
            else:
                L[i,j] = k0 + 10 * a2 * np.log10(d) - CONST
    return L

def short_term_fading(T, N, f_c, speed):
    """ Rayleigh fading model via sum of sinusoids
    from "Model of independent Rayleigh faders" by Z. Wu
    """
    t_vec = np.array(range(T)) / 1e3
    a_0 = np.pi / (2*N)
    w_M = 2 * np.pi * f_c * speed / 3e8
    
    N0 = N // 4
    alpha = a_0 + 2 * np.pi * np.array(range(N0)) / N
    theta = 2 * np.pi * np.random.uniform(0, 1, (N0, 1))
    theta_p = 2 * np.pi * np.random.uniform(0, 1, (N0, 1))
    
    I = np.cos(w_M * np.outer(np.cos(alpha) , t_vec) + np.matlib.repmat(theta, 1, len(t_vec)))
    Q = np.sin(w_M * np.outer(np.sin(alpha) , t_vec) + np.matlib.repmat(theta_p, 1, len(t_vec)))
    
    h = sum(I + 1j * Q) / np.sqrt(N0)
    
    return h

def create_channel_matrix_over_time(m, n, T, R):
    # specify transmitter locations
    while True:
        locTx = np.random.uniform(0, R, (m, 2)) - R / 2
        D_TxTx = distance.cdist(locTx, locTx, 'euclidean')
        for Tx in range(m):
            D_TxTx[Tx, Tx] = float('Inf')
        if np.min(D_TxTx) >= min_D_TxTx:
            break
            
    # specify receiver locations
    while True:
        phi = 2 * np.pi * np.random.uniform(n)
        r = np.sqrt(np.random.uniform(low=min_D_TxRx ** 2, high=max_D_TxRx ** 2, size=(n,)))
        locRx = np.clip(locTx + np.stack((r * np.cos(phi), r * np.sin(phi)), axis=1), a_min= -R / 2, a_max=R / 2)
        
        D_TxRx = distance.cdist(locTx, locRx, 'euclidean')
        if np.min(D_TxRx) < min_D_TxRx:
            continue
            
        L = UDN_PL(D_TxRx) + shadowing * np.random.randn(m, n) # Loss matrix in dB
        H_l = np.sqrt(np.power(10, -L / 10)) # large-scale fading matrix
        associations = (H_l == np.max(H_l, axis=0, keepdims=True))
        if min(np.sum(associations, axis=1)) > 0: # each transmitter has at least one associated reciever
            break

    H = np.zeros((m, n, T), dtype=complex)
    for i in range(m):
        for j in range(n):
            H[i, j] = short_term_fading(T, 100, f_c, speed)
            
    H *= np.expand_dims(H_l, axis=2)
    
    return np.abs(H) ** 2, np.abs(H_l) ** 2