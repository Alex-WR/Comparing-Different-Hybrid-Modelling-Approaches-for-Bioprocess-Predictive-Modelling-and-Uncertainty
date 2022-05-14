# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:26:10 2021

@author: Alex
"""

import numpy as np
import torch
from scipy.integrate import odeint

# %% <Import Cache> 
import HYBRID_MODEL_MAIN as MAIN

# %% <Integrate Kinetic Model>

def solve_kinetic(t, Y0, mu_m, k_c, y_sx, beta):
    
    # Ground Truth Model Individual ODEs:   
    dXdt0 = lambda Y, t: MAIN.KN_dXdt(*Y, mu_m, k_c)
    dSdt0 = lambda Y, t: MAIN.KN_dSdt(*Y, mu_m, k_c, y_sx)
    dPdt0 = lambda Y, t: MAIN.KN_dPdt(*Y, beta)
    
    # Solve Ground Truth Overall ODE:
    dYdt = lambda Y, t: np.array([dXdt0(Y, t), dSdt0(Y, t), dPdt0(Y, t)])
    Y = odeint(dYdt, Y0, t)
    return Y

def generate_kinetic(t, Y0X, std, rep, keys, mu_m, k_c, y_sx, beta):
    Y0X = Y0X * rep
    # Iterate over Initial Conditions:
    insilico_data = {}
    for n, Y0 in zip(keys, Y0X):
        
        # Sample Random Parameters:
        rnd = np.random.RandomState(n+1)       
        parameters = {'mu_m' : rnd.normal(mu_m, std * mu_m),
                      'k_c'  : rnd.normal(k_c, std * k_c),
                      'y_sx' : rnd.normal(y_sx, std * y_sx),
                      'beta' : rnd.normal(beta, std * beta)}
       
        Y = solve_kinetic(t, Y0, **parameters)
        insilico_data[n] = np.column_stack([t, Y])
    return insilico_data

data_key = {'T': 0, 'CX': 1, 'CS': 2, 'CP': 3}

# %% <Integrate Ground Truth Kinetic Model>

def solve_ground_truth(t, Y0, mu_m, k_c, mu_d, y_sx, beta, k_d, k_p):
    
    # Ground Truth Model Individual ODEs:   
    dXdt0 = lambda Y, t: MAIN.GT_dXdt(*Y, mu_m, k_c, mu_d)
    dSdt0 = lambda Y, t: MAIN.GT_dSdt(*Y, mu_m, k_c, y_sx)
    dPdt0 = lambda Y, t: MAIN.GT_dPdt(*Y, beta, k_d, k_p)
    
    # Solve Ground Truth Overall ODE:
    dYdt = lambda Y, t: np.array([dXdt0(Y, t), dSdt0(Y, t), dPdt0(Y, t)])
    Y = odeint(dYdt, Y0, t)
    return Y

# %% <Sample and Integrate>

# Estimated Parameter Means:
def generate_insilico_data(t, Y0X, std, rep, keys, mu_m, k_c, mu_d, y_sx, beta, k_d, k_p):
    Y0X = Y0X * rep
    # Iterate over Initial Conditions:
    insilico_data = {}
    for n, Y0 in zip(keys, Y0X):
        
        # Sample Random Parameters:
        rnd = np.random.RandomState(n+1)       
        parameters = {'mu_m' : rnd.normal(mu_m, std * mu_m),
                      'k_c'  : rnd.normal(k_c, std * k_c),
                      'mu_d' : rnd.normal(mu_d, std * mu_d),
                      'y_sx' : rnd.normal(y_sx, std * y_sx),
                      'beta' : rnd.normal(beta, std * beta),
                      'k_d'  : rnd.normal(k_d, std * k_d),
                      'k_p'  : rnd.normal(k_p, std * k_p)}

        Y = solve_ground_truth(t, Y0, **parameters)
        insilico_data[n] = np.column_stack([t, Y])
    return insilico_data

data_key = {'T': 0, 'CX': 1, 'CS': 2, 'CP': 3}
    
# %% <Insilico Datasets>

# Time Span:
t = np.arange(0, 168 + 12, 12)

# Ground Truth Parameters:
GT_parameters = {'mu_m' : 0.43,
                 'k_c'  : 63.7,
                 'mu_d' : 2.10e-3,
                 'y_sx' : 2.58,
                 'beta' : 0.236,
                 'k_d'  : 6.48e-2,
                 'k_p'  : 2.50}

data_train = generate_insilico_data(t, MAIN.Y0_train, std=MAIN.train_std, rep=MAIN.train_rep, keys=MAIN.train_test_split['train'], **GT_parameters)
data_test = generate_insilico_data(t, MAIN.Y0_test, std=0, rep=MAIN.test_rep, keys=MAIN.train_test_split['test'], **GT_parameters)
data = {**data_train, **data_test}

# %% <Export Cache>
torch.save(GT_parameters, MAIN.cache + '\_GT_parameters')
torch.save(data_train, MAIN.cache + '\_insilico_data_train')
torch.save(data_test, MAIN.cache + '\_insilico_data_test')
torch.save(data, MAIN.cache + '\_insilico_data')
torch.save(data_key, MAIN.cache + '\_data_key')

