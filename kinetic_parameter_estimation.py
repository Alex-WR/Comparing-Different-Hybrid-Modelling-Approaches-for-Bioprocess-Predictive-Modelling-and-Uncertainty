# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:35:00 2021

@author: Alex
"""

from pyomo.environ import *
from pyomo.dae import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from itertools import product
import numpy as np
import torch

# %% <Import Cache> 
import HYBRID_MODEL_MAIN as MAIN
import insilico_data_generation as gdt
GT_parameters = torch.load(MAIN.cache + '\_GT_parameters')
insilico_data = torch.load(MAIN.cache + '\_insilico_data_train')
data_key = torch.load(MAIN.cache + '\_data_key')

# %% <Define Parametric Model>

model = m = AbstractModel()
m.t   = ContinuousSet()
m.e   = Set(dimen=1)

# Measured State Variables:
m.t_meas = Set(within=m.t)
m.X_meas = Param(m.t_meas, m.e)
m.S_meas = Param(m.t_meas, m.e)
m.P_meas = Param(m.t_meas, m.e)

# Initial State Variables:
m.X_init = Param(m.e)
m.S_init = Param(m.e)
m.P_init = Param(m.e)

# State Variables:
m.X = Var(m.t, m.e, within=PositiveReals)
m.S = Var(m.t, m.e, within=PositiveReals)
m.P = Var(m.t, m.e, within=PositiveReals)

# Mean State Variables:
m.X_mean = Param(m.e)
m.S_mean = Param(m.e)
m.P_mean = Param(m.e)

# Specific Biochemiacal Reaction Constants to be Estimated:
m.mu_m = Var(bounds=(0, 2))
m.k_c  = Var(bounds=(0, 500))
m.y_sx = Var(bounds=(0, 10))
m.beta = Var(bounds=(0, 1))
    

# Rate of Change of States:
m.dXdt = DerivativeVar(m.X, wrt=m.t)
m.dSdt = DerivativeVar(m.S, wrt=m.t)
m.dPdt = DerivativeVar(m.P, wrt=m.t)

# Relations for the Rate of Change of States:
    
def _xdot (m, i, e): 
      if i == 0: return Constraint.Skip
      return m.dXdt[i, e] == MAIN.KN_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m, m.k_c)
m.dXdtcon = Constraint(m.t, m.e, rule=_xdot)

def _sdot (m, i, e): 
      if i == 0: return Constraint.Skip
      return m.dSdt[i, e] == MAIN.KN_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m, m.k_c, m.y_sx)
m.dSdtcon = Constraint(m.t, m.e, rule=_sdot)

def _pdot (m, i, e): 
      if i == 0: return Constraint.Skip
      return m.dPdt[i, e] == MAIN.KN_dPdt(m.X[i, e], m.S[i, e], m.P[i, e], m.beta)
m.dPdtcon = Constraint(m.t, m.e, rule=_pdot)

# Define Initial Conditions:
def _init_X(m, e):
    return m.X[0, e] == m.X_init[e]
m.init_X = Constraint(m.e, rule=_init_X)
def _init_S(m, e):
    return m.S[0, e] == m.S_init[e]
m.init_S = Constraint(m.e, rule=_init_S)
def _init_P(m, e):
    return m.P[0, e] == m.P_init[e]
m.init_P = Constraint(m.e, rule=_init_P)

# %% <Initialise Parametric Model>

exps = list(insilico_data.keys())
mean = {e: x.mean(axis=0) for e, x in insilico_data.items()}
t_st = [int(t) for t in insilico_data[1][:, data_key['T']]]

# Initialse Abstract Model:
def initialize_instance():
    data_init = {None: {'t': {None: t_st},
                        't_meas': {None: t_st}, 
                        'e': {None: exps},
                        'X_meas': {(t, e): float(insilico_data[e][i, data_key['CX']]) for e, (i, t) in product(exps, enumerate(t_st))},
                        'X_mean': {e: float(mean[e][data_key['CX']]) for e in exps},
                        'X_init': {e: float(insilico_data[e][0, data_key['CX']]) for e in exps},
                        'S_meas': {(t, e): float(insilico_data[e][i, data_key['CS']]) for e, (i, t) in product(exps, enumerate(t_st))},
                        'S_mean': {e: float(mean[e][data_key['CS']]) for e in exps for e in exps},
                        'S_init': {e: float(insilico_data[e][0, data_key['CS']]) for e in exps},
                        'P_meas': {(t, e): float(insilico_data[e][i, data_key['CP']]) for e, (i, t) in product(exps, enumerate(t_st))},
                        'P_mean': {e: float(mean[e][data_key['CP']]) for e in exps for e in exps},
                        'P_init': {e: float(insilico_data[e][0, data_key['CP']]) for e in exps}}}   

    return model.create_instance(data_init)

# Define Objective Function:  
def _obj(m):
    return sum(sum(((m.X[i, e] - m.X_meas[i, e]) / m.X_mean[e]) ** 2 for i in m.t_meas) +
               sum(((m.S[i, e] - m.S_meas[i, e]) / m.S_mean[e]) ** 2 for i in m.t_meas) +
               sum(((m.P[i, e] - m.P_meas[i, e]) / m.P_mean[e]) ** 2 for i in m.t_meas) for e in m.e) / len(exps)
m.obj = Objective (rule=_obj, sense=minimize)

# %% <Descritise and Solve>
    
# Initialise:
instance = initialize_instance()

# Define Descritization Method:
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(instance, nfe=14, ncp=6, scheme='LAGRANGE-RADAU')

# Call in a solver:
solver = SolverFactory ('ipopt')
solver.options['max_iter'] = 3000
results = solver.solve(instance, tee=True)

# %% <Retrieve Results>

import matplotlib as mpl
size = 20

# Retrieve Kinetic Parameters:
kinetic_parameters = {'mu_m' : value(instance.mu_m),
                      'k_c'  : value(instance.k_c),
                      'y_sx' : value(instance.y_sx),
                      'beta' : value(instance.beta)}

print('mu_m : {:.4f}'.format(kinetic_parameters['mu_m']))
print('k_c  : {:.2f}'.format(kinetic_parameters['k_c']))
print('y_sx : {:.2f}'.format(kinetic_parameters['y_sx']))
print('beta : {:.3f}'.format(kinetic_parameters['beta']))

# Assembling Data Dictionary:
kinetic_data = {}
for e in exps:
    t = []; x = []; s = []; p = []; x_meas = []; s_meas = []; p_meas = []
    for time in sorted(instance.t_meas):
        t.append(time)
        x.append(value(instance.X[time, e]))
        x_meas.append(value(instance.X_meas[time, e]))
        
        s.append(value(instance.S[time, e]))
        s_meas.append(value(instance.S_meas[time, e]))
        
        p.append(value(instance.P[time, e]))
        p_meas.append(value(instance.P_meas[time, e]))
    
    # Solve Hybrid ODE with Estimated Parameters:
    Y0 = [value(instance.X_init[e]), value(instance.S_init[e]), value(instance.P_init[e])]
    Y1 = gdt.solve_kinetic(t, Y0, **kinetic_parameters)
    kinetic_data[e] = np.column_stack([t, Y1])
    Y2 = gdt.solve_ground_truth(t, Y0, **GT_parameters)

    fig = plt.figure(figsize=(12 * size / 10, 4 * size / 10))
    ms  = size / 1.7
    lw  = size / 6
    mpl.rc('font', **{'size'   : size * 1.2})
    mpl.rcParams['xtick.major.size']  = size / 3
    mpl.rcParams['xtick.major.width'] = size / 8
    mpl.rcParams['ytick.major.size']  = size / 3
    mpl.rcParams['ytick.major.width'] = size / 8
    mpl.rcParams['axes.linewidth']    = size / 10
    mpl.rcParams['figure.constrained_layout.use'] = True
    
    plt.subplot(1, 3, 1)
    plt.plot(t, x, '-', color='#E06669', label='Fitted Kinetic Model', ms=ms, lw=lw)
    plt.plot(t, x_meas, 'ko', label='In-Silico Experiment', ms=ms, lw=lw)
    plt.plot(t, x_meas, 'k-', label='Parameter Estimation', ms=ms, lw=lw) # !!!
    plt.plot(t, Y1[:, 0], 'r--', label='Re-Estimated', ms=ms, lw=lw)
    plt.plot(t, Y2[:, 0], 'k--', label='Ground Truth', ms=ms, lw=lw)
    plt.xlabel("Time (h)")
    plt.ylabel("Biomass Concentration (g/L)")
    plt.legend(loc="best")
    
    plt.subplot(1, 3, 2)
    plt.plot(t, s, '-', color='#E06669', label='Fitted Kinetic Model', ms=ms, lw=lw)
    plt.plot(t, s_meas, 'ko', label='In-Silico Experiment', ms=ms, lw=lw)
    plt.plot(t, s_meas, 'k-', label='Parameter Estimation', ms=ms, lw=lw) # !!!
    plt.plot(t, Y1[:, 1], 'r--', label='Re-Estimated', ms=ms, lw=lw)
    plt.plot(t, Y2[:, 1], 'k--', label='Ground Truth', ms=ms, lw=lw)
    plt.xlabel("Time (h)")
    plt.ylabel("Glucose Concentration (g/L)")
    plt.legend(loc="best") 
    
    plt.subplot(1, 3, 3)
    plt.plot(t, p, '-', color='#E06669', label='Fitted Kinetic Model', ms=ms, lw=lw)
    plt.plot(t, p_meas, 'ko', label='In-Silico Experiment', ms=ms, lw=lw)
    plt.plot(t, p_meas, 'k-', label='Parameter Estimation', ms=ms, lw=lw) # !!!
    plt.plot(t, Y1[:, 2], 'r--', label='Re-Estimated', ms=ms, lw=lw)
    plt.plot(t, Y2[:, 2], 'k--', label='Ground Truth', ms=ms, lw=lw)
    plt.xlabel("Time (h)")
    plt.ylabel("Astaxanthin Concentration (mg/L)")
    plt.legend(loc="best")
    
    plt.show()
    
   
# %% <Export Cache>
torch.save(kinetic_data, MAIN.cache + '\_kinetic_data')  
torch.save(kinetic_parameters, MAIN.cache + '\_kinetic_parameters')
torch.save(data_key, MAIN.cache + '\_data_key')