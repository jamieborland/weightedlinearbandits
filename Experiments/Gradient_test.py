#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:03:40 2025

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
    
    # Extract the final cumulative regret for the policy that was tested
    # We use str(policy) as the key to avoid the errors we saw earlier
    final_cumulative_regret = avgRegret[str(policy_to_test)][-1]
    
    return final_cumulative_regret    
 
#%%
if __name__ == "__main__":

    # === 1. META-LEARNING AND EXPERIMENT PARAMETERS ===
    
    # --- Experiment settings (using shorter runs for speed) ---
    steps = 6000
    step_1 = 6000
    steps_calibration = 6000
    n_mc_per_eval = 100  # Monte Carlo runs for each evaluation. Keep it small for speed.
    d = 2 # Dimension of the problem
    k = 50 # Number of arms available at each step
    alpha = 1
    sigma_noise = 1
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
    learning_rate = 1e-6  # How big of a step to take for gamma. Start small.
    h = 0.0001                # The small perturbation for calculating the finite difference

    # Calculate the theoretical optimal gamma to use as our target benchmark
    # Note: B_T, d, R, etc. are used from the global scope defined earlier in your script
    B_T = get_B_T_smooth(step_1, R, angle_init, angle_end, d)
    theoretical_gamma = 1 - (B_T / (d * steps_calibration))**(2/3)

    # === 2. POLICY INITIALIZATION ===

    # Our learning policy, starting with a different gamma to see it learn
    grad_policy = FiniteGradDLinUCB(d, delta, alpha, lambda_, s, 0.9, 'FiniteGrad-DLinUCB', False, sigma_noise, False)

    # === 3. THE MAIN META-LEARNING LOOP ===
    
    gamma_history = []
    print("--- Starting Meta-Learning Experiment ---")
    print(f"Target theoretical gamma for T=2000: {theoretical_gamma:.4f}")
    print(f"Learning Rate: {learning_rate:.7f}")
    
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

        # 1. Evaluate performance at R(gamma)
        # We use a temporary, new DLinUCB object for a clean evaluation
        temp_policy_gamma = DLinUCB(d, delta, alpha, lambda_, s, current_gamma, 'temp_gamma', False, sigma_noise, False)
        regret1 = run_simulation_for_policy(temp_policy_gamma, **sim_params)
        reward1 = -regret1  # We want to maximize reward, which is minimizing regret

        # 2. Evaluate performance at R(gamma + h)
        temp_policy_gamma_h = DLinUCB(d, delta, alpha, lambda_, s, current_gamma + h, 'temp_gamma_h', False, sigma_noise, False)
        regret2 = run_simulation_for_policy(temp_policy_gamma_h, **sim_params)
        reward2 = -regret2
        
        # 3. Calculate the gradient and update the main policy's gamma
        gradient = (reward2 - reward1) / h
        new_gamma = current_gamma + learning_rate * gradient  # Gradient ASCENT on reward
        grad_policy.update_gamma(new_gamma)

        print(f"Epoch {epoch+1}/{meta_epochs} | Current Gamma: {current_gamma:.4f} | Gradient: {gradient:.2f}")

    # === 4. PLOTTING THE RESULTS ===
    print("--- Experiment Finished. Plotting results. ---")

    plt.figure(figsize=(8, 6))
    plt.plot(gamma_history, marker='o', linestyle='--', label='Learned γ')
    plt.axhline(y=theoretical_gamma, color='r', linestyle='--', label=f'Theoretical Optimal γ')
    plt.xlabel('Meta-Learning Epoch')
    plt.ylabel('Gamma Value')
    plt.title('Convergence of Learned Gamma')
    plt.ylim(0.0, 1.0) # Zoom in on a reasonable range for gamma
    plt.legend()
    plt.grid(True)
    plt.show()

