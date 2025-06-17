#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 17:45:17 2025

@author: jamie
"""
#%%
import os
import torch
import torch.optim as optim
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
from PolicyGradDLinUCB import PolicyGradDLinUCB
from environment_class import Environment
from simulator_class import Simulator
from utils import plot_regret, scatter_abrupt, scatter_smooth, action_check, get_B_T_smooth, detection_sorted, generate_smooth_theta



#%%
def run_policy_grad_episode(policy, d, k, steps, step_1, sigma_noise, R, angle_init, angle_end):
    """
    Runs a single, non-parallel episode for the policy gradient algorithm.

    Returns:
        log_probs (list): The list of log-probabilities of the action taken at each step.
        total_reward (float): The total reward accumulated over the episode.
    """
    
    # Reset the policy and environment for a fresh run
    policy.re_init()
    theta_start = np.array([1, 0])
    mab = Environment(d, theta_start, sigma_noise, False)
    
    log_probs = []
    rewards = []
    
    for t in range(steps):
        # Update the environment to be non-stationary
        mab.theta = generate_smooth_theta(t, step_1, R, angle_init, angle_end, d)
        
        # 1. Select an arm (this will use softmax and store the log_prob)
        available_arms = mab.get_arms(k)
        chosen_arm_index = policy.select_arm(available_arms)
        
        # 2. Store the log probability of the action that was taken
        log_probs.append(policy.log_prob)
        
        # 3. Play the arm and get a reward
        reward, features = mab.play(chosen_arm_index)
        rewards.append(reward)
        
        # 4. Update the policy's internal state
        policy.update_state(features, reward)
        
    # The total reward is simply the sum of all rewards in the episode
    return log_probs, np.sum(rewards)

#%%
if __name__ == "__main__":

    # === 1. EXPERIMENT AND META-LEARNING PARAMETERS ===
    
    # Experiment settings
    steps = 1000
    step_1 = 1000
    
    # Meta-Learning settings
    meta_epochs = 10000     
    learning_rate = 0.01   
    
    d = 2; k = 50; alpha = 1; sigma_noise = 0; R = 1; angle_init = 0; angle_end = np.pi/2
    delta = 0.01; s = 1; lambda_ = 0.1;

    # === 2. POLICY GRADIENT LEARNER INITIALIZATION ===
    
    # We will learn the log of the temperature to ensure it's always positive.
    # We define it as a PyTorch tensor that requires a gradient.
    log_temp = torch.tensor(0.0, requires_grad=True)
    
    # The ADAM optimizer will update this learnable parameter.
    optimizer = optim.Adam([log_temp], lr=learning_rate)
    
    #use a fixed, theoretically good gamma.
    B_T = get_B_T_smooth(steps, R, angle_init, angle_end, d)
    theoretical_gamma = 1 - (B_T / (d * steps))**(2/3)

    # === 3. THE MAIN POLICY GRADIENT (REINFORCE) LOOP ===

    reward_history = []
    temp_history = []
    print("--- Starting Policy Gradient Experiment ---")
    
    for epoch in range(meta_epochs):
        
        current_temp = torch.exp(log_temp) # Get current temp from its log
        temp_history.append(current_temp.item())
        
        # Create a policy instance with the current temperature
        policy = PolicyGradDLinUCB(d, delta, alpha, lambda_, s, theoretical_gamma, 'PG-DLinUCB', False, sigma_noise, False, temp=current_temp)

        # Run a single episode to get the log_probs and total reward
        log_probs, total_reward = run_policy_grad_episode(policy, d, k, steps, step_1, sigma_noise, R, angle_init, angle_end)
        
        # 2. Run the second, independent episode to get the baseline reward
        _, baseline_reward = run_policy_grad_episode(policy, d, k, steps, step_1, sigma_noise, R, angle_init, angle_end)
        
        reward_history.append(total_reward)

        # --- REINFORCE  with b_SELF baseline Algorithm Gradient Calculation & Update ---
        
        #Calculate advantage for baseline
        advantage = total_reward - baseline_reward
        
        policy_loss = []
        for log_p in log_probs:
            # The loss for each step is its negative log-probability, scaled by the final reward.
            policy_loss.append(-log_p * advantage)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward() 
        optimizer.step() 

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{meta_epochs} | Avg Reward (last 10): {np.mean(reward_history[-10:]):.2f} | Temp: {current_temp.item():.4f}")
            
         #--- CONVERGENCE CHECK ---
        if epoch > 100 and epoch % 20 == 0:
            
            # Calculate the average temperature over the last 50 epochs
            recent_avg_temp = np.mean(temp_history[-50:])
            
            # Calculate the average temperature of the 50 epochs before that
            previous_avg_temp = np.mean(temp_history[-100:-50])
            
            # Define a small tolerance. If the temperature hasn't changed by this much, we can stop.
            temp_change_threshold = 0.001 
            
            if abs(recent_avg_temp - previous_avg_temp) < temp_change_threshold:
                print(f"\n--- Convergence detected at epoch {epoch}. Stopping training. ---")
                print(f"Temperature has stabilized at approximately {recent_avg_temp:.4f}.")
                break 

    # === 4. PLOTTING THE RESULTS ===
    print("--- Training Finished ---")
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(reward_history)
    ax1.set_title('Reward per Episode During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    ax2.plot(temp_history)
    ax2.set_title('Learned Temperature Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Softmax Temperature')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

#%%



tex_path = '/Library/TeX/texbin'  
os.environ['PATH'] = os.environ['PATH'] + ':' + tex_path

# Update plot parameters for a professional look
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14
})



# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Reward per Episode ---
ax1.plot(reward_history, 
         color='C0',  # Use standard color cycle
         linewidth=2)
ax1.set_title('Reward per Episode During Training', fontsize=16, pad=10)
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Total Reward', fontsize=14)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax1.tick_params(axis='both', which='major', labelsize=12)

# Remove top and right spines for a cleaner look
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Plot 2: Learned Temperature ---
ax2.plot(temp_history, 
         color='C1', 
         linewidth=2)
ax2.set_title(r'Learned Temperature ($\tau$) Over Time', fontsize=16, pad=10)
ax2.set_xlabel('Epoch', fontsize=14)
ax2.set_ylabel(r'Softmax Temperature ($\tau$)', fontsize=14)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax2.tick_params(axis='both', which='major', labelsize=12)

# Remove top and right spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

