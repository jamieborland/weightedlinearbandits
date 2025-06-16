#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:03:40 2025
A finite difference approach used to show convergence to to theoretical optimal gamma. 
This is a first step in exemplifying the use of gradient based methods in linear bandits for non stationary environments.


@author: jamie
"""

#%%
import os
import sys
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the main project root)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)
import matplotlib.pyplot as plt

# Add these lines to disable TeX and use a standard font
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
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


def run_simulation_for_policy(policy_to_test, n_mc, d, k, steps, step_1, sigma_noise, R, angle_init, angle_end, q, t_saved, verbose=False):
    """
    Runs a full simulation for a *single* specified policy and returns its final cumulative regret.
    This function is self-contained and takes all experiment parameters as arguments.
    """
    # Define the starting conditions for the environment for this run
    theta_start = np.array([1, 0])
    bp = {} # No abrupt breakpoints in this smooth environment
    
    # only test the single policy we provide
    policies_to_run = [policy_to_test]
    
    # Create fresh Environment and Simulator instances for this specific run
    mab = Environment(d, theta_start, sigma_noise, verbose)
    simulator = Simulator(mab, theta_start, policies_to_run, k, d, steps, bp, verbose)
    
    # Run the simulation using the existing multiprocessing function
    avgRegret, _, _ = simulator.run_multiprocessing_smooth(n_process=4, step_1=step_1, steps=steps, 
                                                             n_mc=n_mc, R=R, angle_init=angle_init, 
                                                             angle_end=angle_end, q=q, t_saved=t_saved)
    
    # Extract the final cumulative regret for the po}licy that was tested
    # We use str(policy) as the key to avoid the errors we saw earlier
    final_cumulative_regret = avgRegret[str(policy_to_test)][-1]
    
    return final_cumulative_regret
def calculate_gradient(policy_to_update, sim_params, h):
    """
    Calculates the gradient of the reward with respect to gamma using a robust
    finite difference method that handles boundaries.

    It uses a centered difference when possible, and a one-sided difference
    at the boundaries of the [0, 1] interval.
    """
    current_gamma = policy_to_update.gamma
    
    # We must ensure both points we test are valid
    gamma_plus_h = current_gamma + h
    gamma_minus_h = current_gamma - h

    # To be safe, we run both simulations with the same seed.
    seed = int(time.time()) # Use a different seed for each gradient calculation
    
    # Case 1: We are too close to the upper boundary (gamma=1)
    if gamma_plus_h > 1.0:
        print("    (Using backward difference at upper boundary)")
        # --- Calculate reward at gamma ---
        np.random.seed(seed)
        policy1 = DLinUCB(d, delta, alpha, lambda_, s, current_gamma, 'temp1', False, sigma_noise, False)
        reward1 = -run_simulation_for_policy(policy1, **sim_params)
        
        # --- Calculate reward at gamma - h ---
        np.random.seed(seed)
        policy2 = DLinUCB(d, delta, alpha, lambda_, s, gamma_minus_h, 'temp2', False, sigma_noise, False)
        reward2 = -run_simulation_for_policy(policy2, **sim_params)
        
        # Backward difference gradient
        gradient = (reward1 - reward2) / h
        
    # Case 2: We are too close to the lower boundary (gamma=0)
    elif gamma_minus_h < 0.0:
        print("    (Using forward difference at lower boundary)")
        # --- Calculate reward at gamma + h ---
        np.random.seed(seed)
        policy1 = DLinUCB(d, delta, alpha, lambda_, s, gamma_plus_h, 'temp1', False, sigma_noise, False)
        reward1 = -run_simulation_for_policy(policy1, **sim_params)
        
        # --- Calculate reward at gamma ---
        np.random.seed(seed)
        policy2 = DLinUCB(d, delta, alpha, lambda_, s, current_gamma, 'temp2', False, sigma_noise, False)
        reward2 = -run_simulation_for_policy(policy2, **sim_params)

        # Forward difference gradient
        gradient = (reward1 - reward2) / h
        
    # Case 3: We are safely in the middle
    else:
        # --- Calculate reward at gamma + h ---
        np.random.seed(seed)
        policy1 = DLinUCB(d, delta, alpha, lambda_, s, gamma_plus_h, 'temp1', False, sigma_noise, False)
        reward1 = -run_simulation_for_policy(policy1, **sim_params)
        
        # --- Calculate reward at gamma - h ---
        np.random.seed(seed)
        policy2 = DLinUCB(d, delta, alpha, lambda_, s, gamma_minus_h, 'temp2', False, sigma_noise, False)
        reward2 = -run_simulation_for_policy(policy2, **sim_params)
        
        # Centered difference gradient (more accurate)
        gradient = (reward1 - reward2) / (2 * h)
        
    return gradient    
 
#%%
if __name__ == "__main__":

    # === 1. META-LEARNING AND EXPERIMENT PARAMETERS ===
    
    # --- Experiment settings (using shorter runs for speed) ---
    steps = 6000
    step_1 = 6000
    steps_calibration = 6000
    n_mc_per_eval = 10 # Monte Carlo runs for each evaluation. Keep it small for speed.
    d = 2 # Dimension of the problem
    k = 50 # Number of arms available at each step
    alpha = 1
    sigma_noise = 0
    verbose = False
    q = 5 # 5 percent quantiles used
    R = 1 # True parameter evolving on the unit circle
    t_saved = None # Saving the entire trajectory
    delta = 0.01 # Probability of being outside the confidence interval
    s = 1  # Bound on the theta_star parameter
    lambda_ = 0.1 # Regularisation parameter
    q = 5 # Diplaying the quantile (in %)

    angle_init = 0
    angle_end = np.pi/2

    # --- Meta-Learning settings ---
    meta_epochs = 100         # How many times we will update gamma
    learning_rate = 15e-6  # How big of a step to take for gamma. Start small.
    h = 0.01  # The small perturbation for calculating the finite difference

    # Calculate the theoretical optimal gamma to use as our target benchmark
    # Note: B_T, d, R, etc. are used from the global scope defined earlier in the script
    B_T = get_B_T_smooth(step_1, R, angle_init, angle_end, d)
    theoretical_gamma = 1 - (B_T / (d * steps_calibration))**(2/3)

    # === 2. POLICY INITIALIZATION ===

    # Our learning policy, starting with a different gamma to see it learn
    grad_policy = FiniteGradDLinUCB(d, delta, alpha, lambda_, s, 0.5, 'FiniteGrad-DLinUCB', False, sigma_noise, False)

    # === 3. THE MAIN META-LEARNING LOOP ===
    
    gamma_history = []
    print("--- Starting Meta-Learning Experiment ---")
    print(f"Target theoretical gamma for T={steps} steps is {theoretical_gamma:.4f}")
    print(f"Learning Rate: {learning_rate:.7f}")
    print(f"Monte Carlo runs: {n_mc_per_eval}")
    
    for epoch in range(meta_epochs):
        
        current_gamma = grad_policy.gamma
        gamma_history.append(current_gamma)

        # --- Finite Differences Gradient Calculation ---
        # We create a dictionary of all the parameters our simulation function needs
        # This makes the function call clean and robust
        sim_params = {
            'n_mc': n_mc_per_eval, 'd': d, 'k': k, 'steps': steps, 'step_1': step_1,
            'sigma_noise': sigma_noise, 'R': R, 'angle_init': angle_init,
            'angle_end': angle_end, 'q': q, 't_saved': t_saved
        }
        # Use a different seed for each epoch, but the same seed for both reward calculations within the epoch.
        epoch_seed = epoch 

        gradient = calculate_gradient(grad_policy, sim_params, h) # All the logic is now in the helper function

        new_gamma = current_gamma + learning_rate * gradient  # Gradient ASCENT on reward
        grad_policy.update_gamma(new_gamma)

        print(f"Epoch {epoch+1}/{meta_epochs} | Current Gamma: {current_gamma:.4f} | Gradient: {gradient:.2f}")
        
#%%
#Set path for latex (issue for me running in spyder otherwise)
tex_path = '/Library/TeX/texbin' 
os.environ['PATH'] = os.environ['PATH'] + ':' + tex_path

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], 
    "font.size": 14  
})


fig, ax = plt.subplots(figsize=(7, 5)) 

# Plot the learned gamma with markers every 5 epochs
ax.plot(gamma_history, 
        marker='o',          
        linestyle='-',      
        linewidth=2,        
        markersize=6,    
        markevery=5,         
        label='Learned $\\gamma$') 

# Plot the theoretical gamma line
ax.axhline(y=theoretical_gamma, 
           color='r', 
           linestyle='--', 
           linewidth=1.5, 
           label=f'Theoretical Optimal $\\gamma$ ({theoretical_gamma:.4f})')

# --- Labels, Title, and Ticks ---
ax.set_xlabel('Meta-Learning Epoch', fontsize=14)
ax.set_ylabel('Gamma Value ($\\gamma$)', fontsize=14)
ax.set_title('Convergence of Learned Gamma', fontsize=16, pad=10)
ax.tick_params(axis='both', which='major', labelsize=12) # Control tick label size

# --- Aesthetics ---
ax.set_ylim(0.5, 1.01) # Set y-limits
ax.legend(loc='lower right', fontsize=12) # Control legend location and size
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7) # Lighter grid

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
