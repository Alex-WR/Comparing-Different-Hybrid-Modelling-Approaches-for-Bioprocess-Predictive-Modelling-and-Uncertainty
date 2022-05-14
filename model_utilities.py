# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 09:33:52 2022

@author: Alex
"""

import numpy as np
import pandas as pd
from copy import deepcopy                        

import GPy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# %% <Import Cache>
import HYBRID_MODEL_MAIN as MAIN
torch.manual_seed(MAIN.rnd_seed) 

# %% <Artifical Neural Network>

class ANN(nn.Module):
    def __init__(self, hyprams):
        super().__init__()
        
        # Defining ANN Topology:
        self.input_size   = hyprams['input_size']
        self.hidden_size  = hyprams['hidden_size']
        self.output_size  = hyprams['output_size']
        
        # Defining Activation Function:
        self.activation = nn.Tanh()

        # Define Layers:
        sizes = [self.input_size, *self.hidden_size, self.output_size]
        self.linears = nn.ModuleList([nn.Linear(i, j) for i, j in zip(sizes[:-1], sizes[1:])])

    def forward(self, x):
        x = x.float()
        for linear in self.linears[:-1]:
            x = linear(x)
            x = self.activation(x)
        x = self.linears[-1](x)
        return x * 2
    
    def init_params(self):
        for linear in self.linears:
            torch.nn.init.xavier_normal_(linear.weight.data)
            torch.nn.init.zeros_(linear.bias)

# %% <ANN Training>           
            
class ANN_trainer(object):
    def __init__(self, hyprams, x_index, u_index, y_index):
        self.dtype = torch.float
        self.hyprams = hyprams
        self.x_index = x_index
        self.u_index = u_index
        self.y_index = y_index
        
    def fit(self, data_train, data_valid):
        
        # Reshape, Convert Numpy Array to PyTorch Tensors and Split X and Y Variables:
        f = lambda x1: np.row_stack(np.transpose(x1, axes=(2, 0, 1)))
        g = lambda x2: np.row_stack([f(x1) for x1 in x2.values()])
        data_train = torch.from_numpy(g(data_train)).type(self.dtype)
        self.x_train = data_train[:, self.x_index]
        self.y_train = data_train[:, self.u_index]
        data_valid = torch.from_numpy(g(data_valid)).type(self.dtype)
        self.x_valid = data_valid[:, self.x_index]
        self.y_valid = data_valid[:, self.u_index]
        
        # Set Input and Output Layer Sizes from X and Y Sizes:
        self.hyprams['input_size'] = len(self.x_index)
        self.hyprams['output_size'] = len(self.u_index)
        
        # Initialize ANN:
        self.ANN = ANN(self.hyprams)
        self.ANN.init_params()
        
        # Initialize Gradient Decent Optimizer and Loss Function:
        self.optimizer = optim.Adam(self.ANN.parameters(), lr=self.hyprams['learning_rate'])
        self.loss_fn = nn.MSELoss()
        
        # Iterate to Minimise Loss Function:
        self.training_history = np.zeros((self.hyprams['epochs'], 3))
        for epoch in range(self.hyprams['epochs']):
        
            # Reset Gradients:
            self.optimizer.zero_grad()
            
            # Forward Propagation:
            y_train_pred = self.ANN(self.x_train)
            
            # Compute Training Loss:
            training_loss = self.loss_fn(self.y_train, y_train_pred)
            
            # Backward Propagation:
            training_loss.backward()
            
            # Gradient Descent Step:
            self.optimizer.step()
            
            # Predict Validation:
            y_valid_pred = self.ANN(self.x_valid)
            
            # Compute Validation Loss:
            validation_loss = self.loss_fn(self.y_valid, y_valid_pred)
            
            # Record Training and Validation Loss:
            self.training_history[epoch, 0] = epoch
            self.training_history[epoch, 1] = training_loss
            self.training_history[epoch, 2] = validation_loss
            
    def fit_aggregate(self, data_trainval):
        self.aggregate_ANN = []
        training_history = []
        for k in data_trainval.keys():
            data_train = {i: x for i, x in data_trainval.items() if i is not k}
            data_valid = {k: data_trainval[k]}
            self.fit(data_train, data_valid)
            self.aggregate_ANN.append(deepcopy(self.ANN))
            training_history.append(self.training_history)
        self.training_history = np.mean(training_history, axis=0)

    def predict(self, x):
        y_aggregate = [ANN(x) for ANN in self.aggregate_ANN]
        y_aggregate = torch.dstack(y_aggregate)
        y_mean = torch.mean(y_aggregate, dim=2)
        y_std = torch.std(y_aggregate, dim=2)
        return y_mean, y_std
    
# %% <Gaussian Process Fitting>

class GP_trainer(object):
    def __init__(self, hyprams, x_index, u_index, y_index):
        self.dtype = torch.float
        self.hyprams = hyprams
        self.x_index = x_index
        self.u_index = u_index
        self.y_index = y_index
        
    def fit(self, data_train):
        
        # Extract X and Y:
        f = lambda x1: np.row_stack(np.transpose(x1, axes=(2, 0, 1)))
        g = lambda x2: np.row_stack([f(x1) for x1 in x2.values()])
        data_train = g(data_train)
        self.x = data_train[:, self.x_index]
        
        # Set Input and Output Layer Sizes from X and Y Sizes:
        self.input_size = len(self.x_index)
        self.output_size = len(self.u_index)
        
        # Define and Optimise GPs:
        self.GPs = []; 
        for var in self.u_index:
            self.y = data_train[:, var].reshape(-1, 1)
            kernel = GPy.kern.Exponential(input_dim=self.input_size, lengthscale=1, variance=1)
            GP = GPy.models.GPRegression(self.x, self.y, kernel, noise_var=1)
            GP.optimize()
            self.GPs.append(GP)
        
    def predict(self, x):
        y_mean = np.zeros((x.shape[0], self.output_size))
        y_std = deepcopy(y_mean)
        for i, GP in enumerate(self.GPs):
            y_mean[:, i], y_std[:, i] = GP.predict(x)
        return y_mean, y_std
    
    
    
    
    
    
    
    