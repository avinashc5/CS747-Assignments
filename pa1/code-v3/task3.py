"""
Task 3: Optimized KL-UCB Implementation

This file implements both standard and optimized KL-UCB algorithms for multi-armed bandits.
The optimized version aims to reduce computational overhead while maintaining good regret performance.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Base Algorithm Class ------------------

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# ------------------ KL-UCB utilities ------------------
## You can define other helper functions here if needed

def kl_bern(p, q, eps=1e-12):
    """KL divergence between Bernoulli(p) and Bernoulli(q)."""
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def kl_ucb_index(emp_mean, n, t, c=3.0, tol=1e-6):
    """Compute KL-UCB index with exponential + binary search."""
    if n == 0:
        return 1.0
    
    log_t = np.log(t)
    log_log_t = np.log(max(log_t, 1.0000001))
    
    beta = (log_t + c * log_log_t) / n
    lo, hi = emp_mean, 1.0
    
    # Binary search refinement
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if kl_bern(emp_mean, mid) > beta:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

class KL_UCB_Optimized(Algorithm):
    """
    Optimized KL-UCB algorithm with batching.
    Uses exponential + binary search for safe pulls of the current best arm.
    """
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.num_arms = num_arms
        self.horizon = horizon
        self.counts = np.zeros(num_arms, dtype=int)
        self.successes = np.zeros(num_arms, dtype=float)
        self.t = 0
        self.c = 3
        self.indices = np.ones(num_arms, dtype=float)  # start optimistic
        self.last_pulled = -1
        self.batch_remaining = 0
        self.batch_size = 1

    def _update_indices(self):
        """
        Update the KL-UCB indices for all arms.
        """
        for i in range(self.num_arms):
            if self.counts[i] > 0:
                emp_mean = self.successes[i] / self.counts[i]
                self.indices[i] = kl_ucb_index(emp_mean, self.counts[i], self.t, self.c)
            else:
                self.indices[i] = 1.0  # optimistic init

    def give_pull(self):
        """
        Select an arm to pull based on the current state of the algorithm.
        
        Returns:
        int: Index of the arm to pull (0-indexed).
        """
        # Warm-up: play each arm once
        if self.t < self.num_arms:
            self.last_pulled = self.t
            return self.last_pulled

        # Continue current batch
        if self.batch_remaining > 0:
            self.batch_remaining -= 1
            return self.last_pulled

        # Update and pick new best
        self._update_indices()
        if self.last_pulled != np.argmax(self.indices):
            self.last_pulled = np.argmax(self.indices)
            self.batch_size = 1  # reset batch size on arm change
            self.batch_remaining = self.batch_size - 1
        else:
            self.batch_size = min(self.batch_size * 2, 1000)
            self.batch_remaining = self.batch_size - 1

        self.c = 3.0 * (1 - self.t / self.horizon)  # decay c over time
        return self.last_pulled

    def get_reward(self, arm_index, reward):
        """
        Update algorithm's internal state based on the arm that was pulled and the reward that was received.
        
        Parameters:
        arm_index (int): Index of the arm that was pulled (0-indexed).
        reward (float): Reward received after pulling the arm.
        """
        self.t += 1
        self.counts[arm_index] += 1
        self.successes[arm_index] += reward

# ------------------ Bonus KL-UCB Algorithm (Optional - 1 bonus mark) ------------------

class KL_UCB_Bonus(Algorithm):
    """
    BONUS ALGORITHM (Optional - 1 bonus mark)
    
    This algorithm must produce EXACTLY IDENTICAL regret trajectories to KL_UCB_Standard
    while achieving significant speedup. Students implementing this will earn 1 bonus mark.
    
    Requirements for bonus:
    - Must produce identical regret trajectories (checked with strict tolerance)
    - Must achieve specified speedup thresholds on bonus testcases
    - Must include detailed explanation in report
    """
    # You can define other functions also in the class if needed

    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        pass
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        pass
        #END EDITING HERE
