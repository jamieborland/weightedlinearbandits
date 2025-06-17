#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 17:30:38 2025

@author: jamie
"""
import torch
import numpy as np
from D_LinUCB_class import DLinUCB # Make sure this import works

class PolicyGradDLinUCB(DLinUCB):
    """
    A D-LinUCB policy that uses a softmax function for action selection,
    making it differentiable for policy gradient methods.
    """
    def __init__(self, d, delta, alpha, lambda_, s, gamma, name, sm, sigma_noise, verbose, temp=1.0):
        # Call the parent constructor
        super().__init__(d, delta, alpha, lambda_, s, gamma, name, sm, sigma_noise, verbose)
        
        # A new hyperparameter for the softmax function, controlling exploration.
        self.temperature = temp
        
        # This will store the log-probability of the chosen action, needed for the gradient update.
        self.log_prob = None

    def select_arm(self, arms):
        """
        Selects an arm using a softmax distribution over the UCB scores.
        This version uses PyTorch to enable automatic differentiation.
        """
        k_t = len(arms)
        ucb_s = np.zeros(k_t)
    
        const1 = np.sqrt(self.lambda_) * self.s
        beta_t = const1 + self.sigma_noise * \
                 np.sqrt(self.c_delta + self.dim * np.log(1 + (1 - self.gamma2_t) / (self.dim *
                                                  self.lambda_ * (1 - self.gamma**2))))
        for (i, a) in enumerate(arms):
            a_features = a.features
            invcov_a = np.inner(self.invcov @ self.cov_squared @ self.invcov, a_features.T)
            ucb_s[i] = np.dot(self.hat_theta, a_features) + self.alpha * beta_t * np.sqrt(np.dot(a_features, invcov_a))
        
        
        # 1. Convert NumPy scores to a PyTorch tensor
        ucb_s_tensor = torch.from_numpy(ucb_s)
        
        # 2. Apply temperature and calculate softmax probabilities
        # self.temperature is now also a tensor passed from our main loop
        probabilities = torch.nn.functional.softmax(ucb_s_tensor / self.temperature, dim=0)
    
        # 3. Create a probability distribution and sample an action
        # This creates a stochastic policy
        dist = torch.distributions.Categorical(probabilities)
        chosen_arm_index = dist.sample()
        
        # 4. Store the log-probability of the chosen action for the gradient update later
        self.log_prob = dist.log_prob(chosen_arm_index)
        
        return np.int_(chosen_arm_index.item())

    def __str__(self):
        return f'{self.name}' # A simpler name for this class
        
    @staticmethod
    def id():
        return 'PolicyGrad-DLinUCB'