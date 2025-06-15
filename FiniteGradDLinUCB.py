#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:14:46 2025

@author: jamie
"""

from D_LinUCB_class import DLinUCB
import numpy as np

class FiniteGradDLinUCB(DLinUCB):
    """
    A D-LinUCB policy designed to work with a meta-learner that updates
    the gamma parameter. It inherits all core logic from DLinUCB.
    """
    def __init__(self, d, delta, alpha, lambda_, s, initial_gamma, name, sm, sigma_noise, verbose):
        """
        Initializes the policy.
        
        Args:
            initial_gamma (float): The starting value for the discount factor, gamma.
            All other arguments are the same as the parent DLinUCB class.
        """
        # Call the parent constructor to set up all the standard attributes.
        # It will use 'initial_gamma' to set the starting self.gamma.
        super().__init__(d, delta, alpha, lambda_, s, initial_gamma, name, sm, sigma_noise, verbose)

    def update_gamma(self, new_gamma):
        """
        Allows an external process to update the gamma parameter.
        
        Args:
            new_gamma (float): The new value for the discount factor.
        """
        # Keep gamma within a reasonable, stable range to prevent divergence.
        self.gamma = np.clip(new_gamma, 0.0, 0.999999)

    def __str__(self):
        """
        Overrides the default string representation for clearer logging and plotting.
        """
        # The name attribute is inherited from the parent class.
        return f'{self.name} (γ={self.gamma:.4f})'
    
    @staticmethod
    def id():
        # This helps in identifying the policy in the simulator's results dictionary.
        return 'FiniteGrad-DLinUCB'