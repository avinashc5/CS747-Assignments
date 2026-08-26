# simulator
# do not modify! (except final few lines)

from bernoulli_bandit import *
from typing import List
import numpy as np
from numpy.random import SeedSequence, default_rng
from task1 import Algorithm, Eps_Greedy, UCB, KL_UCB, Thompson_Sampling
from task2 import PoissonDoorsEnv, StudentPolicy, Policy

from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
import sys
import os
  
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliBandit

def single_sim(seed=0, ALGO=Algorithm, PROBS=[0.3, 0.5, 0.7], HORIZON=1000):
  np.random.seed(seed)
  shuffled_probs = np.random.permutation(PROBS)
  bandit = BernoulliBandit(probs=shuffled_probs)
  algo_inst = ALGO(num_arms=len(shuffled_probs), horizon=HORIZON)
  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
  return bandit.regret()

def run_episode(env: PoissonDoorsEnv, policy: Policy, max_steps: int = 100_000) -> int:
    env.reset()
    if hasattr(policy, "reset_stats"):
        policy.reset_stats()
    steps = 0
    for t in range(1, max_steps + 1):
        arm = policy.select_arm(t)
        reward, done, info = env.step(arm)
        if hasattr(policy, "update") and policy.update.__code__.co_argcount >= 4:
            policy.update(arm, reward, info["health"])
        else:
            policy.update(arm, reward)
        steps += 1
        if done:
            break
    return steps

def simulate_task2_once(mus: List[float], H0: int, env_rng, pol_rng, max_steps: int) -> int:
    env = PoissonDoorsEnv(mus, H0=H0, rng=env_rng)
    policy = StudentPolicy(env.K, rng=pol_rng)
    return run_episode(env, policy, max_steps=max_steps)

def simulate_task2(
    mus: List[float],
    H0: int = 100,
    master_seed: int = 1729,
    max_steps: int = 100_000,
    episodes: int = 200,
) -> int | float:
    ss_master = SeedSequence(master_seed)

    if episodes <= 1:
        ss_env, ss_pol = ss_master.spawn(2)
        env_rng = default_rng(ss_env)
        pol_rng = default_rng(ss_pol)
        return simulate_task2_once(mus, H0, env_rng, pol_rng, max_steps)

    total = 0
    for _ in range(episodes):
        ss_env, ss_pol = ss_master.spawn(2)
        env_rng = default_rng(ss_env)
        pol_rng = default_rng(ss_pol)
        total += simulate_task2_once(mus, H0, env_rng, pol_rng, max_steps)
    return total / episodes

def single_sim_task3(seed=0, ALGO=Algorithm, PROBS=[0.3, 0.5, 0.7], HORIZON=1000):
  """Single simulation run for task 3 with timing measurement."""
  np.random.seed(seed)
  shuffled_probs = np.random.permutation(PROBS)
  bandit = BernoulliBandit(probs=shuffled_probs)
  
  start_time = time.time()
  algo_inst = ALGO(num_arms=len(shuffled_probs), horizon=HORIZON)
  
  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
  
  end_time = time.time()
  execution_time = end_time - start_time
  
  return bandit.regret(), execution_time

def simulate(algorithm, probs, horizon, num_sims=50):
  """simulates algorithm of class Algorithm
  for BernoulliBandit bandit, with horizon=horizon
  """
  
  def multiple_sims(num_sims=3):
    with Pool(10) as pool:
      sim_out = pool.starmap(single_sim,
        [(i, algorithm, probs, horizon) for i in range(num_sims)])
    return sim_out 

  sim_out = multiple_sims(num_sims)
  regrets = np.mean(sim_out)

  return regrets

def simulate_task3(algorithm, probs, horizon, num_sims=50):
  """simulates algorithm for task 3 and returns average regret and time
  """
  
  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      sim_out = pool.starmap(single_sim_task3,
        [(i, algorithm, probs, horizon) for i in range(num_sims)])
    return sim_out 

  sim_out = multiple_sims(num_sims)
  regrets = [result[0] for result in sim_out]
  times = [result[1] for result in sim_out]
  
  avg_regret = np.mean(regrets)
  avg_time = np.mean(times)

  return avg_regret, avg_time 

def task1(algorithm, probs, num_sims=50):
  """generates the plots and regrets for task1
  """
  horizons = [2**i for i in range(10, 19)]
  regrets = []
  for horizon in horizons:
    regrets.append(simulate(algorithm, probs, horizon, num_sims))

  print(regrets)
  plt.plot(horizons, regrets)
  plt.title("Regret vs Horizon")
  plt.savefig("task1-{}.png".format(algorithm.__name__))
  plt.clf()

def task3(probs, num_sims=50):
  """Compare standard and optimized KL-UCB algorithms for task3."""
  from task3 import KL_UCB_Optimized
  import matplotlib.pyplot as plt
  
  import importlib
  autograder_module = importlib.import_module('autograder')
  KL_UCB_Standard = autograder_module.KL_UCB_Standard
  
  horizons = [2**i for i in range(10, 15)]
  
  standard_regrets = []
  standard_times = []
  optimized_regrets = []
  optimized_times = []
  
  for horizon in horizons:
    print(f"Testing horizon {horizon}...")
    
    # Test standard KL-UCB
    reg_std, time_std = simulate_task3(KL_UCB_Standard, probs, horizon, num_sims)
    standard_regrets.append(reg_std)
    standard_times.append(time_std)
    
    # Test optimized KL-UCB
    reg_opt, time_opt = simulate_task3(KL_UCB_Optimized, probs, horizon, num_sims)
    optimized_regrets.append(reg_opt)
    optimized_times.append(time_opt)
    
    print(f"  Standard: Regret={reg_std:.2f}, Time={time_std:.3f}s")
    print(f"  Optimized: Regret={reg_opt:.2f}, Time={time_opt:.3f}s")
    print(f"  Speedup: {time_std/time_opt:.2f}x")
  
  # Plot results
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  
  # Plot regrets
  ax1.plot(horizons, standard_regrets, 'o-', label='Standard KL-UCB', linewidth=2)
  ax1.plot(horizons, optimized_regrets, 's-', label='Optimized KL-UCB', linewidth=2)
  ax1.set_xlabel('Horizon')
  ax1.set_ylabel('Average Regret')
  ax1.set_title('Regret Comparison')
  ax1.legend()
  ax1.grid(True, alpha=0.3)
  ax1.set_xscale('log')
  
  ax2.plot(horizons, standard_times, 'o-', label='Standard KL-UCB', linewidth=2)
  ax2.plot(horizons, optimized_times, 's-', label='Optimized KL-UCB', linewidth=2)
  ax2.set_xlabel('Horizon')
  ax2.set_ylabel('Average Execution Time (s)')
  ax2.set_title('Execution Time Comparison')
  ax2.legend()
  ax2.grid(True, alpha=0.3)
  ax2.set_xscale('log')
  ax2.set_yscale('log')
  
  plt.tight_layout()
  plt.savefig(f"task3_comparison.png")
  plt.show()
  
  return {
    'horizons': horizons,
    'standard_regrets': standard_regrets,
    'standard_times': standard_times,
    'optimized_regrets': optimized_regrets,
    'optimized_times': optimized_times
  }

def task3_bonus_trajectories(probs, horizon=5000, num_sims=3):
  """Compare trajectories of standard and bonus KL-UCB algorithms."""
  try:
    from task3 import KL_UCB_Bonus
    import importlib
    autograder_module = importlib.import_module('autograder')
    KL_UCB_Standard = autograder_module.KL_UCB_Standard
  except ImportError:
    print("KL_UCB_Bonus not implemented - skipping bonus trajectory comparison")
    return None
  
  import matplotlib.pyplot as plt
  from bernoulli_bandit import BernoulliBandit
  
  print(f"Testing bonus algorithm trajectory similarity...")
  
  base_seed = 42
  trajectories_standard = []
  trajectories_bonus = []
  
  for sim in range(num_sims):
    seed = base_seed + sim
    
    np.random.seed(seed)
    shuffled_probs = np.random.permutation(probs)
    bandit = BernoulliBandit(probs=shuffled_probs)
    algo_inst = KL_UCB_Standard(num_arms=len(shuffled_probs), horizon=horizon)
    
    trajectory_std = []
    for t in range(horizon):
      arm_to_be_pulled = algo_inst.give_pull()
      reward = bandit.pull(arm_to_be_pulled)
      algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
      trajectory_std.append(bandit.regret())
    
    np.random.seed(seed)
    shuffled_probs = np.random.permutation(probs)
    bandit = BernoulliBandit(probs=shuffled_probs)
    algo_inst = KL_UCB_Bonus(num_arms=len(shuffled_probs), horizon=horizon)
    
    trajectory_bonus = []
    for t in range(horizon):
      arm_to_be_pulled = algo_inst.give_pull()
      reward = bandit.pull(arm_to_be_pulled)
      algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
      trajectory_bonus.append(bandit.regret())
    
    trajectories_standard.append(trajectory_std)
    trajectories_bonus.append(trajectory_bonus)
  
  max_diff = 0.0
  all_identical = True
  
  for sim in range(num_sims):
    for t in range(horizon):
      diff = abs(trajectories_standard[sim][t] - trajectories_bonus[sim][t])
      max_diff = max(max_diff, diff)
      if diff > 1e-6:
        all_identical = False
  
  print(f"Trajectory analysis:")
  print(f"  Maximum difference: {max_diff:.2e}")
  print(f"  Trajectories identical: {all_identical}")
  
  fig, axes = plt.subplots(2, 2, figsize=(15, 10))
  
  for sim in range(min(num_sims, 3)):
    axes[0, 0].plot(trajectories_standard[sim], alpha=0.7, label=f'Standard Sim {sim+1}')
    axes[0, 1].plot(trajectories_bonus[sim], alpha=0.7, label=f'Bonus Sim {sim+1}')
  
  axes[0, 0].set_title('Standard KL-UCB Trajectories')
  axes[0, 0].set_xlabel('Time Step')
  axes[0, 0].set_ylabel('Cumulative Regret')
  axes[0, 0].legend()
  axes[0, 0].grid(True, alpha=0.3)
  
  axes[0, 1].set_title('Bonus KL-UCB Trajectories')
  axes[0, 1].set_xlabel('Time Step')
  axes[0, 1].set_ylabel('Cumulative Regret')
  axes[0, 1].legend()
  axes[0, 1].grid(True, alpha=0.3)
  
  for sim in range(min(num_sims, 3)):
    diff_trajectory = [abs(trajectories_standard[sim][t] - trajectories_bonus[sim][t]) 
                      for t in range(horizon)]
    axes[1, 0].plot(diff_trajectory, alpha=0.7, label=f'Diff Sim {sim+1}')
  
  axes[1, 0].set_title('Absolute Difference Between Trajectories')
  axes[1, 0].set_xlabel('Time Step')
  axes[1, 0].set_ylabel('|Standard - Bonus|')
  axes[1, 0].legend()
  axes[1, 0].grid(True, alpha=0.3)
  axes[1, 0].set_yscale('log')
  
  axes[1, 1].plot(trajectories_standard[0], 'b-', label='Standard', linewidth=2)
  axes[1, 1].plot(trajectories_bonus[0], 'r--', label='Bonus', linewidth=2, alpha=0.8)
  axes[1, 1].set_title('Trajectory Overlay (First Simulation)')
  axes[1, 1].set_xlabel('Time Step')
  axes[1, 1].set_ylabel('Cumulative Regret')
  axes[1, 1].legend()
  axes[1, 1].grid(True, alpha=0.3)
  
  plt.tight_layout()
  plt.savefig(f"task3_bonus_trajectories.png")
  plt.show()
  
  return {
    'trajectories_identical': all_identical,
    'max_difference': max_diff,
    'trajectories_standard': trajectories_standard,
    'trajectories_bonus': trajectories_bonus
  }

if __name__ == '__main__':
  ### EDIT only the following code ###

  # TASK 1 STARTS HERE
  # Note - all the plots generated for task 1 will be for the following 
  # bandit instance:
  # 20 arms with uniformly distributed means

  task1probs = [i/10 for i in range(1,10)]
  task1(Eps_Greedy, task1probs, 1)
  # task1(UCB, task1probs)
  # task1(KL_UCB, task1probs)
  # task1(Thompson_Sampling, task1probs)
  # TASK 1 ENDS HERE

  # TASK 2 STARTS HERE
  # mus = [1.08, 1.2]   ### change this to test on custom testcases
  # steps = simulate_task2(mus, H0=100, master_seed=1729)  # you can change the seed here, but for Task 2 autograder evaluation we are using a fixed seed: 1729 to ensure reproducibility

  # print(f"Average number of steps: {steps}")
  # TASK 2 ENDS HERE
  
  # TASK 3 STARTS HERE
  # Test with a simple bandit instance for task 3
  # task3probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  # task3(task3probs, num_sims=10)  # Reduced sims for faster testing
  
  # BONUS: Test trajectory similarity for bonus algorithm (uncomment to test)
  # task3_bonus_trajectories(task3probs, horizon=5000, num_sims=3)
  # TASK 3 ENDS HERE
