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
insilico_data = torch.load(MAIN.cache + '\_insilico_data') # !!!
I = torch.load(MAIN.cache + '\_data_key')


# %% <Define Parametric Model>
model = m = AbstractModel()
m.t   = ContinuousSet()
m.e   = Set(dimen=1)

# Time Intervals:
t_step = 12
i_step = range(insilico_data[1].shape[0] - 1)

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
m.k_c  = Var(bounds=(0, 100))
m.y_sx = Var(within=PositiveReals)
for i in i_step:
    setattr(m, 'mu_m' + str(i), Var(m.e, within=Reals))
    setattr(m, 'beta' + str(i), Var(m.e, within=Reals))

# Rate of Change of States:
m.dXdt = DerivativeVar(m.X, wrt=m.t)
m.dSdt = DerivativeVar(m.S, wrt=m.t)
m.dPdt = DerivativeVar(m.P, wrt=m.t)


def _xdot (m, i, e): 
      if i == 0:
         return Constraint.Skip
      elif i>0 and i<=12:
         return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m0[e], m.k_c)
      elif i>12 and i<=24:
         return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m1[e], m.k_c)
      elif i>24 and i<=36:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m2[e], m.k_c)
      elif i>36 and i<=48:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m3[e], m.k_c)
      elif i>48 and i<=60:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m4[e], m.k_c)
      elif i>60 and i<=72:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m5[e], m.k_c)
      elif i>72 and i<=84:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m6[e], m.k_c)
      elif i>84 and i<=96:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m7[e], m.k_c)
      elif i>96 and i<=108:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m8[e], m.k_c)
      elif i>108 and i<=120:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m9[e], m.k_c)
      elif i>120 and i<=132:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m10[e], m.k_c)
      elif i>132 and i<=144:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m11[e], m.k_c)
      elif i>144 and i<=156:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m12[e], m.k_c)  
      elif i>156 and i<=168:
          return m.dXdt[i, e] == MAIN.HY_dXdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m13[e], m.k_c)
      
m.dXdtcon = Constraint (m.t, m.e, rule=_xdot)

def _sdot (m, i, e): 
      if i == 0:
         return Constraint.Skip
      elif i>0 and i<=12:
         return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m0[e], m.k_c, m.y_sx)
      elif i>12 and i<=24:
         return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m1[e], m.k_c, m.y_sx)
      elif i>24 and i<=36:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m2[e], m.k_c, m.y_sx)
      elif i>36 and i<=48:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m3[e], m.k_c, m.y_sx)
      elif i>48 and i<=60:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m4[e], m.k_c, m.y_sx)
      elif i>60 and i<=72:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m5[e], m.k_c, m.y_sx)
      elif i>72 and i<=84:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m6[e], m.k_c, m.y_sx)
      elif i>84 and i<=96:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m7[e], m.k_c, m.y_sx)
      elif i>96 and i<=108:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m8[e], m.k_c, m.y_sx)
      elif i>108 and i<=120:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m9[e], m.k_c, m.y_sx)
      elif i>120 and i<=132:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m10[e], m.k_c, m.y_sx)
      elif i>132 and i<=144:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m11[e], m.k_c, m.y_sx)
      elif i>144 and i<=156:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m12[e], m.k_c, m.y_sx)  
      elif i>156 and i<=168:
          return m.dSdt[i, e] == MAIN.HY_dSdt(m.X[i, e], m.S[i, e], m.P[i, e], m.mu_m13[e], m.k_c, m.y_sx)
      
m.dSdtcon = Constraint(m.t, m.e, rule=_sdot)

def _pdot (m, i, e): 
      if i == 0:
         return Constraint.Skip
      elif i>0 and i<=12:
         return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta0[e])
      elif i>12 and i<=24:
         return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta1[e])
      elif i>24 and i<=36:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta2[e])
      elif i>36 and i<=48:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta3[e])
      elif i>48 and i<=60:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta4[e])
      elif i>60 and i<=72:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta5[e])
      elif i>72 and i<=84:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta6[e])
      elif i>84 and i<=96:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta7[e])
      elif i>96 and i<=108:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta8[e])
      elif i>108 and i<=120:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta9[e])
      elif i>120 and i<=132:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta10[e])
      elif i>132 and i<=144:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta11[e])
      elif i>144 and i<=156:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta12[e])  
      elif i>156 and i<=168:
          return m.dPdt[i, e] == MAIN.HY_dPdt(m.X[i, e], m.S[i,e], m.P[i, e], m.beta13[e])
      
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
t_st = [int(t) for t in insilico_data[1][:, I['T']]]

def initialize_instance():
    data_init = {None: {'t': {None: t_st},
                        't_meas': {None: t_st}, 
                        'e': {None: exps},
                        'X_meas': {(t, e): float(insilico_data[e][i, I['CX']]) for e, (i, t) in product(exps, enumerate(t_st))},
                        'X_mean': {e: float(mean[e][I['CX']]) for e in exps},
                        'X_init': {e: float(insilico_data[e][0, I['CX']]) for e in exps},
                        'S_meas': {(t, e): float(insilico_data[e][i, I['CS']]) for e, (i, t) in product(exps, enumerate(t_st))},
                        'S_mean': {e: float(mean[e][I['CS']]) for e in exps for e in exps},
                        'S_init': {e: float(insilico_data[e][0, I['CS']]) for e in exps},
                        'P_meas': {(t, e): float(insilico_data[e][i, I['CP']]) for e, (i, t) in product(exps, enumerate(t_st))},
                        'P_mean': {e: float(mean[e][I['CP']]) for e in exps for e in exps},
                        'P_init': {e: float(insilico_data[e][0, I['CP']]) for e in exps}}}   

    return model.create_instance(data_init)

# Define Objective Function: 
weights = [0, 0]
varbs   = ['mu_m', 'beta']
    
def _obj(m):
    error =  sum(sum(((m.X[i, e] - m.X_meas[i, e]) / m.X_mean[e]) ** 2 for i in m.t_meas) +
                 sum(((m.S[i, e] - m.S_meas[i, e]) / m.S_mean[e]) ** 2 for i in m.t_meas) +
                 sum(((m.P[i, e] - m.P_meas[i, e]) / m.P_mean[e]) ** 2 for i in m.t_meas) for e in m.e) / len(exps)
       
    regularisation = 0
# =============================================================================
#     for var, weight in zip(varbs, weights):
#         regularisation += weight * sum((sum((((getattr(m, var + str(i+1))[e] - getattr(m, var + str(i))[e])) ** 2) for i in i_step[:-1]) for e in exps))
#     regularisation /= len(varbs)
# =============================================================================
    
    return error + regularisation

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

# %% <Retrieve Reslts>

import matplotlib as mpl
size = 20

# Retrieve Kinetic Parameters:
I = {'T': 0, 'CX': 1, 'CS': 2, 'CP': 3, 'mu_m': 4, 'k_c': 5, 'y_sx': 6, 'beta': 7}
parameterised_data = {e: np.column_stack([insilico_data[e],
                         np.append(np.array([value(getattr(instance, 'mu_m' + str(i))[e]) for i in i_step]), np.nan),
                         np.append(np.array([value(getattr(instance, 'k_c')) for i in i_step]), np.nan),
                         np.append(np.array([value(getattr(instance, 'y_sx')) for i in i_step]), np.nan),
                         np.append(np.array([value(getattr(instance, 'beta' + str(i))[e]) for i in i_step]), np.nan)])
                      for e in insilico_data.keys()}


print('k_c : {:.4f}'.format(value(instance.k_c)))
print('y_sx : {:.4f}'.format(value(instance.y_sx)))

ms  = size / 1.7
lw  = size / 6
mpl.rc('font', **{'size'   : size * 1.2})
mpl.rcParams['xtick.major.size']  = size / 3
mpl.rcParams['xtick.major.width'] = size / 8
mpl.rcParams['ytick.major.size']  = size / 3
mpl.rcParams['ytick.major.width'] = size / 8
mpl.rcParams['axes.linewidth']    = size / 10
mpl.rcParams['figure.constrained_layout.use'] = True

# Assembling Data Dictionary:
for e in insilico_data.keys():
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
    Y = np.zeros((len(t), 3))
    Y[0, :] = insilico_data[e][0, 1:]
    for i in range(Y.shape[0]-1):
        mu_m = parameterised_data[e][i, I['mu_m']]
        k_c  = parameterised_data[e][i, I['k_c']]
        y_sx = parameterised_data[e][i, I['y_sx']]
        beta = parameterised_data[e][i, I['beta']]
        dXdt0 = lambda Y, t: MAIN.HY_dXdt(*Y, mu_m, k_c)
        dSdt0 = lambda Y, t: MAIN.HY_dSdt(*Y, mu_m, k_c, y_sx)
        dPdt0 = lambda Y, t: MAIN.HY_dPdt(*Y, beta)
        dYdt = lambda Y, t: np.array([dXdt0(Y, t), dSdt0(Y, t), dPdt0(Y, t)])
        Y[i:i+2, :] = odeint(dYdt, Y[i, :], [t[i], t[i+1]])

    
    #Graphing estimated vs actual Exp1 data
    fig = plt.figure(figsize=(12 * size / 10, 4 * size / 10))
    
    plt.subplot(1, 3, 1)
    plt.plot(t, x, '-', color='#E06669', label='Fitted Hybrid Model', ms=ms, lw=lw)
    plt.plot(t, x_meas, 'ko', label='In-Silico Experiment', ms=ms, lw=lw)
    plt.plot(t, Y[:, 0], 'r--', label='Re-Integrated', ms=ms, lw=lw)
    plt.xlabel("Time(h)")
    plt.ylabel("Biomass concentration (g/L)")
    plt.legend(loc="best")
    
    plt.subplot(1, 3, 2)
    plt.plot(t, s, '-', color='#E06669', label='Fitted Hybrid Model', ms=ms, lw=lw)
    plt.plot(t, s_meas, 'ko', label='In-Silico Experiment', ms=ms, lw=lw)
    plt.plot(t, Y[:, 1], 'r--', label='Re-Integrated', ms=ms, lw=lw)
    plt.xlabel("Time(h)")
    plt.ylabel("Glucose concentration (g/L)")
    plt.legend(loc="best") 
    
    plt.subplot(1, 3, 3)
    plt.plot(t, p, '-', color='#E06669', label='Fitted Hybrid Model', ms=ms, lw=lw)
    plt.plot(t, p_meas, 'ko', label='In-Silico Experiment', ms=ms, lw=lw)
    plt.plot(t, Y[:, 2], 'r--', label='Re-Integrated', ms=ms, lw=lw)
    plt.xlabel("Time(h)")
    plt.ylabel("Astaxanthin concentration (mg/L)")
    plt.legend(loc="best")
    
    plt.show()
    
# %% <Export Cache>
torch.save(parameterised_data, MAIN.cache + '\_parameterised_data')  
torch.save(I, MAIN.cache + '\_data_key')   