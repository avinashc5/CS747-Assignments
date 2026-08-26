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




# ------------------ Optimized KL-UCB Algorithm ------------------

# class KL_UCB_Optimized(Algorithm):
#     """
#     Optimized KL-UCB algorithm that reduces computation while maintaining identical regret.
#     This implements a batched KL-UCB with exponential+binary search for safe pulls of the current best arm.
#     """
#     ## You can define other functions also in the class if needed
    
#     def __init__(self, num_arms, horizon):
#         super().__init__(num_arms, horizon)
#         # can initialize member variables here
#         # START EDITING HERE
#         self.counts = np.zeros(num_arms, dtype=int)
#         self.successes = np.zeros(num_arms, dtype=float)
#         self.t = 0
#         self.c = 3.0
#         self.indices = np.zeros(num_arms, dtype=float)
#         self.last_pulled = -1
#         #END EDITING HERE
    
#     def give_pull(self):
#         #START EDITING HERE
#         self.t += 1
        
#         if self.t <= self.num_arms:
#             self.indices[self.t - 1] = 1.0
#             self.last_pulled = self.t - 1
#             return self.last_pulled
        
#         if self.t % 20 != 0:
#             # Pull the last pulled arm to reduce computation
#             return self.last_pulled

#         # if the best arm has greater kl ucb than the updated kl ucb second best arm, pull the best arm
#         if self.indices[np.argmax(self.indices)] >= kl_ucb_index(self.successes[np.argmax(self.indices)] / np.maximum(self.counts[np.argmax(self.indices)], 1), self.counts[np.argmax(self.indices)], self.t, c=self.c):
#             # Safe to pull the best arm
#             self.last_pulled = np.argmax(self.successes / np.maximum(self.counts, 1))
#         else:
#             self.last_pulled = np.argmax(self.indices)
            
#         return self.last_pulled
#         #END EDITING HERE
    
#     def get_reward(self, arm_index, reward):
#         #START EDITING HERE
#         self.counts[arm_index] += 1
#         self.successes[arm_index] += reward
#         self.indices[arm_index] = kl_ucb_index(self.successes[arm_index] / self.counts[arm_index], self.counts[arm_index], self.t, c=self.c)
        #END EDITING HERE
        

# class KL_UCB_Optimized(Algorithm):
#     """
#     Optimized KL-UCB algorithm that reduces computation while maintaining identical regret.
#     This implements a batched KL-UCB with exponential+binary search for safe pulls of the current best arm.
#     """
#     ## You can define other functions also in the class if needed
    
#     def __init__(self, num_arms, horizon):
#         super().__init__(num_arms, horizon)
#         # can initialize member variables here
#         # START EDITING HERE
#         self.counts = np.zeros(num_arms, dtype=int)
#         self.successes = np.zeros(num_arms, dtype=float)
#         self.t = 0
#         self.c = 0.2
#         self.indices = np.zeros(num_arms, dtype=float)
#         self.last_pulled = -1
#         self.batch_remaining = 0
#         #END EDITING HERE
        
#     def _update_indices(self):
#         # vectorized update of all indices
#         for i in range(self.num_arms):
#             if self.counts[i] > 0:
#                 emp_mean = self.successes[i] / self.counts[i]
#                 self.indices[i] = kl_ucb_index(emp_mean, self.counts[i], self.t, self.c)
#             else:
#                 self.indices[i] = 1.0
    
#     def give_pull(self):
#         #START EDITING HERE
#         if self.t < self.num_arms:
#             self.last_pulled = self.t
#             return self.last_pulled
        
#         if self.batch_remaining > 0:
#             self.batch_remaining -= 1
#             return self.last_pulled
        
#         self._update_indices()
#         self.last_pulled = int(np.argmax(self.indices))
        
#         self.batch_remaining = int(np.log2(self.t))
#         return self.last_pulled
#         #END EDITING HERE
    
#     def get_reward(self, arm_index, reward):
#         #START EDITING HERE
#         self.t += 1
#         self.counts[arm_index] += 1
#         self.successes[arm_index] += reward
#         # END EDITING HERE
        

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

    def _update_indices(self):
        for i in range(self.num_arms):
            if self.counts[i] > 0:
                emp_mean = self.successes[i] / self.counts[i]
                self.indices[i] = kl_ucb_index(emp_mean, self.counts[i], self.t, self.c)
            else:
                self.indices[i] = 1.0  # optimistic init

    def safe_batch_exponential(self, best):
        # Compute competitor index = max of all others
        competitors = [j for j in range(self.num_arms) if j != best]
        competitor_index = max(self.indices[j] for j in competitors)

        emp_best = self.successes[best] / self.counts[best]

        # 1) Exponential search
        step = 1
        while step <= self.horizon - self.t:
            new_count = self.counts[best] + step
            new_time = self.t + step
            new_index = kl_ucb_index(emp_best, new_count, new_time, self.c)
            if new_index < competitor_index:
                break
            step *= 2

        # Search interval
        L = step // 2
        R = min(step, self.horizon - self.t)

        # 2) Binary search
        best_ok = L
        while L <= R:
            mid = (L + R) // 2
            new_count = self.counts[best] + mid
            new_time = self.t + mid
            new_index = kl_ucb_index(emp_best, new_count, new_time, self.c)
            if new_index >= competitor_index:
                best_ok = mid
                L = mid + 1
            else:
                R = mid - 1

        return best_ok

    def give_pull(self):
        if self.num_arms == 2:
            return self.t % 2  # alternate for 2 arms
        
        
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
        self.last_pulled = int(np.argmax(self.indices))

        # Horizon-aware batch size
        max_remaining = self.horizon - self.t
        safe_batch = min(int(np.log2(self.t)), max_remaining)
        self.batch_remaining = max(0, safe_batch - 1)
        self.c = 3.0 * (1 - self.t / self.horizon)  # decay c over time
        return self.last_pulled

    def get_reward(self, arm_index, reward):
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
