import numpy as np
import math
from typing import List, Optional, Dict, Tuple

# =========================================================
# ===============   ENVIRONMENT (Poisson)   ===============
# =========================================================

class PoissonDoorsEnv:
    """
    This creates a Poisson environment. There are K doors and each has an associated mean.
    In each step you pick an arm i. Damage to a door is drawn from its corresponding
    Poisson Distribution. Initial health of each door is H0 and decreases by damage in each step.
    Game ends when any door's health < 0.
    """
    def __init__(self, mus: List[float], H0: int = 100, rng: Optional[np.random.Generator] = None):
        self.mus = np.array(mus, dtype=float)
        assert np.all(self.mus > 0), "Poisson means must be > 0"
        self.K = len(mus)
        self.H0 = H0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.health = np.full(self.K, self.H0, dtype=float)
        self.t = 0
        return self.health.copy()

    def step(self, arm: int) -> Tuple[float, bool, Dict]:
        reward = float(self.rng.poisson(self.mus[arm]))
        self.health[arm] -= reward
        self.t += 1
        done = np.any(self.health < 0.0)
        return reward, done, {"reward": reward, "health": self.health.copy(), "t": self.t}


# =========================================================
# =====================   POLICIES   ======================
# =========================================================

class Policy:
    """
    Base Policy interface.
    - Implement select_arm(self, t) to return an int in [0, K-1] to choose an arm.
    - Optionally override update(...) for custom learning.
    """
    def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
        self.K = K
        self.rng = rng if rng is not None else np.random.default_rng()
        self.counts = np.zeros(K, dtype=int)
        self.sums   = np.zeros(K, dtype=float)

    def reset_stats(self):
        self.counts[:] = 0
        self.sums[:]   = 0.0

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.sums[arm] += reward

    @property
    def means(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.sums / np.maximum(self.counts, 1)

    def select_arm(self, t: int) -> int:
        raise NotImplementedError

class StudentPolicy(Policy):
    """
    Thompson Sampling for Poisson Doors.
    
    """
    def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
        super().__init__(K, rng)
        self.alpha = np.ones(K, dtype=float) # The amount of hit points done to each door
        self.beta = np.ones(K, dtype=float) # 
        self.health = np.full(K, 100, dtype=float)
        self.H0 = 100

    def select_arm(self, t: int) -> int:
        sampled_lambdas = self.rng.gamma(self.alpha, 1.0 / (self.beta))
        # sampled_lambdas = self.rng.gamma(self.alpha, 1.0 / (self.beta + np.mean(self.alpha/(self.counts + 1))))
        expected_hits_to_kill = np.where(sampled_lambdas > 0, self.health / sampled_lambdas, float('inf'))
        return int(np.argmin(expected_hits_to_kill))

    def update(self, arm: int, reward: float):
        self.beta[arm] += np.mean(self.alpha/(self.counts + 1)) - (self.alpha[arm] / (self.counts[arm] + 1))  # Increment beta to reflect more data
        self.beta[arm] = max(self.beta[arm], 1e-3)
        # self.beta[arm] += self.alpha[np.argmax(self.alpha/(self.counts + 1))] / (self.counts[np.argmax(self.alpha/(self.counts + 1))] + 1) - (self.alpha[arm] / (self.counts[arm] + 1))  # Increment beta to reflect more data
        # self.beta[arm] += 1.0
        super().update(arm, reward)
        self.alpha[arm] += reward
        self.health[arm] -= reward
        self.health[arm] = max(self.health[arm], 0.0)
