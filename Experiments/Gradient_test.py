#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:03:40 2025

@author: jamie
"""

#%%
import os
os.chdir("..")
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from math import log
import random
import time
import pickle
from numpy.linalg import pinv
from arm_class import ArmGaussian
from LinUCB_class import PolicyLinUCB
from dLinUCB_class import DynamicLinUCB
from D_LinUCB_class import DLinUCB
from SW_LinUCB_class import SWLinUCB
from FiniteGradDLinUCB import FiniteGradDLinUCB
from environment_class import Environment
from simulator_class import Simulator
from utils import plot_regret, scatter_abrupt, scatter_smooth, action_check, get_B_T_smooth, detection_sorted

#%%
# General parameters for all the experiments
delta = 0.01 # Probability of being outside the confidence interval
s = 1  # Bound on the theta_star parameter
lambda_ = 0.1 # Regularisation parameter
q = 5 # Diplaying the quantile (in %)
#%%
def experiment_smooth(n_mc, option):
    mab = Environment(d, theta, sigma_noise, verbose)
    simulator = Simulator(mab, theta, policies, k, d, steps, {}, verbose)
    if option == "par":
        print('n_process:', n_process)
        print('Step1:', step_1)
        print("steps:", steps)
        print("n_mc:", n_mc)
        print("Angle Init:", angle_init)
        print("Angle End:", angle_end)
        print("q:", q)
        avgRegret, qRegret, QRegret = simulator.run_multiprocessing_smooth(n_process, step_1, steps, 
                                                                           n_mc, R, angle_init, angle_end,
                                                                           q, t_saved)
        return avgRegret, qRegret, QRegret
    else:
        avgRegret, qRegret, QRegret, timedic, theta_true, theta_hat = simulator.run_smooth_environment(step_1, steps, 
                                                                                n_mc, q, R, angle_init, angle_end,
                                                                                n_scat, n_scat_true,
                                                                                t_saved)
        return avgRegret, qRegret, QRegret, timedic, theta_true, theta_hat

def data_from_experiment_smooth(n_mc, option):
    if option == "par":
        avgRegret, qRegret, QRegret = experiment_smooth(n_mc, option)
        data = [[policy, avgRegret[policy], qRegret[policy],
                QRegret[policy]] for policy in avgRegret]
        return data
    else:    
        avgRegret, qRegret, QRegret, timedic, theta_true, theta_hat = experiment_smooth(n_mc, option)
        data = [[policy, avgRegret[policy], qRegret[policy],
                QRegret[policy]] for policy in avgRegret]
        return data, theta_hat, theta_true, timedic
#%%
# The saved files should be located in the saved/ folder
# The output images would be located in the out/ folder
# If the folders are not created they must be created to run the code without bugs

path = os.getcwd()
out_dir = 'out_SLOW_NEW'
saved_data_path = 'saved_SLOW_NEW'
detection_folder = 'detection_out_SLOW_NEW'

if not os.path.exists(path + '/' + out_dir):
    os.mkdir(path + '/' + out_dir)
    print('Creating the folder %s' %out_dir)
else:
    print("%s already exists" %out_dir)
    
if not os.path.exists(path + '/' + saved_data_path):
    os.mkdir(path + '/' + saved_data_path)
    print('Creating the folder %s' %saved_data_path)
else:
    print("%s already exists" %saved_data_path)
    
if not os.path.exists(path + '/' + detection_folder):
    os.mkdir(path + '/' + detection_folder)
    print('Creating the folder %s' %detection_folder)
else:
    print("%s already exists" %detection_folder)
#%%
def save_file(filename, var):
    with open(saved_data_path + '/' + str(filename) + '.pkl', 'wb') as f:
        pickle.dump(var, f)
def load_file(filename):
    with open(saved_data_path + '/' + str(filename) + '.pkl', 'rb') as f:
        res = pickle.load(f)
        return res

def save_file_from_folder(folder, filename, var):
    with open(str(folder) + '/' + str(filename) + '.pkl', 'wb') as f:
        pickle.dump(var, f)
def load_file_from_folder(folder, filename):
    with open(str(folder) + '/' + str(filename) + '.pkl', 'rb') as f:
        res = pickle.load(f)
        return res    
#%%
# PARAMETERS
d = 2 # Dimension of the problem
k = 50 # Number of arms available at each step

# Steps_part
step_1 = 3000 # number of steps for the smooth modification
steps = 6000 # for the scatter plot only -> total number of steps
steps_calibration = 6000


n_scat = 400
n_scat_true = 300
t_saved = None # Saving the entire trajectory

# The following commented lines allow to save only some points on the trajectory
# number_t_saved = steps//10
# t_saved = np.int_(np.linspace(0, steps - 1, number_t_saved))

alpha = 1
sigma_noise = 1

verbose = False
q = 5 # 5 percent quantiles used
R = 1 # True parameter evolving on the unit circle

angle_init = 0
angle_end = np.pi/2

B_T = get_B_T_smooth(step_1, R, angle_init, angle_end, d)
print('B_T value:', B_T)
print('Sigma value for the experimenxt:', sigma_noise)

theta = np.array([1,  0]) # Starting point of the unknown parameter
bp = {} # No breakpoints but continuous changes

gamma  = 1 - (B_T/(d*steps_calibration))**(2/3) # Optimal Value to minimize the asymptotical regret
tau = (d*steps_calibration/B_T)**(2/3) # Optimal Value to minimize the asymptotical regret

    
policies = [DLinUCB(d, delta, alpha, lambda_, s, gamma, '', sm = False, sigma_noise = sigma_noise, verbose=verbose),
            FiniteGradDLinUCB(d, delta, alpha, lambda_, s, gamma, '', sm = False, sigma_noise = sigma_noise, verbose=verbose)
           ]    
#%%
# Small experiment with 2 repetitions
data_2, hat_2, true_2, time_2 = data_from_experiment_smooth(n_mc=2, option = '')