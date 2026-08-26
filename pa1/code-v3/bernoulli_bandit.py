# BernoulliArm and BernoulliBandit
# do not modify!

import numpy as np
import matplotlib.pyplot as plt

class BernoulliArm:
  def __init__(self, p):
    self.p = p

  def pull(self, num_pulls=None):
    return np.random.binomial(1, self.p, num_pulls)

class BernoulliBandit:
  def __init__(self, probs=[0.3, 0.5, 0.7],):
    self.__arms = [BernoulliArm(p) for p in probs]
    self.__max_p = max(probs)
    self.__regret = 0

  def pull(self, index):
    reward = self.__arms[index].pull()
    self.__regret += self.__max_p - self.__arms[index].p
    return reward

  def regret(self):
    return self.__regret
  
  def num_arms(self):
    return len(self.__arms)
