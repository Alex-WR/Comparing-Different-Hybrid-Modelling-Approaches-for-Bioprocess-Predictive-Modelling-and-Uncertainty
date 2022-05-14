# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:47:02 2022

@author: Alex
"""
import numpy as np
import pandas as pd
import torch

cache = r'C:\Users\Alex\Google Drive\Documents\University\Projects\RSC Book\Chapter 3\Case Studies\Synthetic Astaxanthin\Cache'

rnd_seed  = 1 # !!! 1
train_std = 0.00
train_rep = 1
test_rep  = 1

# Initial Conditions:
Y0a = [0.1, 10, 0]
Y0b = [0.2, 5, 0]
Y0c = [0.05, 14, 0]
Y0j = [0.1, 12, 0]

Y0d = [0.15, 7.5, 0]
Y0e = [0.05, 5, 0]
Y0f = [0.2, 15, 0]
Y0l = [0.075, 7.5, 0]

Y0g = [0.1, 12, 0]
Y0h = [0.2, 13, 0]
Y0i = [0.05, 14, 0]
Y0k = [0.05, 3, 0]

Y0_train = [Y0a, Y0b, Y0c, Y0j]
Y0_test = [Y0d, Y0e, Y0f, Y0l]

GT_dXdt = lambda X, S, P, mu_m, k_c, mu_d : (mu_m * S /(S + k_c * X)) * X - mu_d * X
GT_dSdt = lambda X, S, P, mu_m, k_c, y_sx : -y_sx * (mu_m * S /(S + k_c * X)) * X
GT_dPdt = lambda X, S, P, beta, k_d, k_p  : beta * X - k_d * X ** 2 * P / (P + k_p)

KN_dXdt = lambda X, S, P, mu_m, k_c       : (mu_m * S /(S + k_c)) * X
KN_dSdt = lambda X, S, P, mu_m, k_c, y_sx : -y_sx * (mu_m * S /(S + k_c)) * X
KN_dPdt = lambda X, S, P, beta            : beta * X

HY_dXdt = lambda X, S, P, mu_m, k_c       : mu_m * (S / (S + k_c)) * X
HY_dSdt = lambda X, S, P, mu_m, k_c, y_sx : - y_sx * mu_m * (S / (S + k_c)) * X
HY_dPdt = lambda X, S, P, beta            : beta * X

w_Y = 1e-6 # 1e-6

train_test_split = {'train': np.arange(len(Y0_train) * train_rep), 
                    'test': len(Y0_train) * train_rep + np.arange(len(Y0_test) * test_rep)}


if __name__ == '__main__':

    from insilico_data_generation import GT_parameters, data_train, data_test
    from hybrid_parameter_estimation import parameterised_data
    from kinetic_parameter_estimation import kinetic_parameters, kinetic_data
    import hybrid_model_type_1_ANN
    import hybrid_model_type_2_ANN
    from hybrid_parameter_estimation_type_3 import parameterised_data
    import hybrid_model_type_3_ANN


