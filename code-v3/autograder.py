import argparse, time
from simulator import simulate, simulate_task2, simulate_task3
from task1 import Algorithm, Eps_Greedy, UCB, KL_UCB, Thompson_Sampling
from task3 import KL_UCB_Optimized
import os
import glob
from multiprocessing import Pool
import time
from bernoulli_bandit import BernoulliBandit
import numpy as np
import math
from task3 import KL_UCB_Bonus


def kl_bern_hidden(p, q, eps=1e-12):
    """KL divergence between Bernoulli(p) and Bernoulli(q)."""
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def kl_ucb_index_hidden(emp_mean, n, t, c=3.0, tol=1e-6, max_iter=25):
    """Compute KL-UCB index for Bernoulli with binary search."""
    if n == 0:
        return 1.0
    beta = (math.log(t) + c * math.log(max(math.log(max(t, 2)), 1.0000001))) / n
    lo, hi = emp_mean, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if kl_bern_hidden(emp_mean, mid) > beta:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

class Testcase:
    def __init__(self, task, probs, horizon):
        self.task = task
        self.probs = probs
        self.horizon = horizon
        self.ucb = 0
        self.kl_ucb = 0
        self.thompson = 0
        self.set_algo = 0
        self.regret_threshold = 0
        self.speedup_threshold = 1.0

def read_tc(path):
    tc = None
    with open(path, 'r') as f:
        lines = f.readlines()
        task = int(lines[0].strip())
        horizon = int(lines[1].strip())
        if task == 1:
            probs = [float(p) for p in lines[2].strip().split()]
            ucb, kl_ucb, thompson = [float(x) for x in lines[3].strip().split()]
            tc = Testcase(task, probs, horizon)
            tc.ucb = ucb
            tc.kl_ucb = kl_ucb
            tc.thompson = thompson
        elif task == 2:
            probs = [float(p) for p in lines[2].strip().split()]
            reference = [float(x) for x in lines[3].strip().split()]
            tc = Testcase(task, probs, horizon)
            tc.set_algo = reference[0]
        elif task == 3:
            probs = [float(p) for p in lines[2].strip().split()]
            thresholds = [float(x) for x in lines[3].strip().split()]
            tc = Testcase(task, probs, horizon)
            tc.regret_threshold = thresholds[0]
            tc.speedup_threshold = thresholds[1]
            
    return tc

def grade_task1(tc_path, algo):
    algo = algo.lower()
    tc = read_tc(tc_path)
    regrets = {}
    scores = {}
    if algo == 'ucb' or algo == 'all':
        regrets['UCB'] = simulate(UCB, tc.probs, tc.horizon)
        scores['UCB'] = 1 if regrets['UCB'] <= tc.ucb else 0
    if algo == 'kl_ucb' or algo == 'all':
        regrets['KL-UCB'] = simulate(KL_UCB, tc.probs, tc.horizon)
        scores['KL-UCB'] = 1 if regrets['KL-UCB'] <= tc.kl_ucb else 0
    if algo == 'thompson' or algo == 'all':
        regrets['Thompson Sampling'] = simulate(Thompson_Sampling, tc.probs, tc.horizon)
        scores['Thompson Sampling'] = 1 if regrets['Thompson Sampling'] <= tc.thompson else 0
    
    return scores, regrets

class KL_UCB_Standard(Algorithm):
    """Hidden standard KL-UCB algorithm implementation for grading purposes."""
    
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms, dtype=int)
        self.successes = np.zeros(num_arms, dtype=float)
        self.t = 0
        self.c = 3.0
    
    def give_pull(self):
        self.t += 1
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        indices = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            emp_mean = self.successes[arm] / self.counts[arm]
            indices[arm] = kl_ucb_index_hidden(emp_mean, self.counts[arm], self.t, self.c)
        
        return int(np.argmax(indices))
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.successes[arm_index] += reward

# ================================================================
# ===================== Task 2 Autograder ========================
# ================================================================

### For Task 2 evaluation of testcases we are using a fixed seed: 1729 to ensure reproducibility
TESTCASE_DIR = "testcases"
H0_DEFAULT = 100
MASTER_SEED = 1729
MAX_STEPS = 100_000

def parse_testcase(path: str):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    task_id = int(lines[0])
    if task_id != 2:
        raise ValueError(f"{path}: First line must be 2 for task 2, got {task_id}.")
    mus = [float(x) for x in lines[1].split()]
    threshold = int(float(lines[2]))
    return mus, threshold

def grade_task2():
    pattern = os.path.join(TESTCASE_DIR, "task2-*.txt")
    files = sorted(glob.glob(pattern), key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("-")[1]))

    if not files:
        print("No task2 testcases found.")
        return

    total = len(files)
    passed = 0

    print(f"=== Autograder: Task 2 ===")
    for path in files:
        try:
            mus, threshold = parse_testcase(path)
            result = simulate_task2(
                mus,
                H0=H0_DEFAULT,
                master_seed=MASTER_SEED,
                max_steps=MAX_STEPS,
            )
            steps = float(result)
            ok = steps < threshold
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            print(f"{os.path.basename(path)}: steps={steps:.1f}  threshold={threshold}  ->  {status}")
        except Exception as e:
            print(f"{os.path.basename(path)}: ERROR - {e}")

    print(f"\nSummary: {passed}/{total} passed")
    return passed

def grade_task3(tc_path):
    """Grade task3 with 3 runs and take the best score."""
    best_result = None
    best_score = -1
    
    print("  Running 3 evaluations and taking the best result...")
    
    for run_num in range(3):
        print(f"    Run {run_num + 1}/3...", end=" ")
        result = _grade_task3_single_run(tc_path)
        print(f"Score: {result['score']}")
        
        if result['score'] > best_score:
            best_score = result['score']
            best_result = result
    
    print(f"  Best score across 3 runs: {best_score}")
    return best_result

def _grade_task3_single_run(tc_path):
    """Single run of task3 grading."""
    tc = read_tc(tc_path)
    
    num_sims = 3
    base_seed = 42
    
    def simulate_with_trajectories(algorithm_class, probs, horizon, num_sims, base_seed):
        """Simulate and return both average metrics and individual trajectories."""
        
        def single_sim_with_trajectory(seed, algorithm_class, probs, horizon):
            """Single simulation returning complete regret trajectory."""
            np.random.seed(seed)
            shuffled_probs = np.random.permutation(probs)
            bandit = BernoulliBandit(probs=shuffled_probs)
            
            start_time = time.time()
            algo_inst = algorithm_class(num_arms=len(shuffled_probs), horizon=horizon)
            
            regret_trajectory = []
            for t in range(horizon):
                arm_to_be_pulled = algo_inst.give_pull()
                reward = bandit.pull(arm_to_be_pulled)
                algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
                regret_trajectory.append(bandit.regret())
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            return regret_trajectory, execution_time
        
        all_trajectories = []
        all_times = []
        
        for i in range(num_sims):
            seed = base_seed + i
            trajectory, exec_time = single_sim_with_trajectory(seed, algorithm_class, probs, horizon)
            all_trajectories.append(trajectory)
            all_times.append(exec_time)
        
        avg_final_regret = sum(traj[-1] for traj in all_trajectories) / num_sims
        avg_time = sum(all_times) / num_sims
        
        return avg_final_regret, avg_time, all_trajectories
    
    standard_regret, standard_time, standard_trajectories = simulate_with_trajectories(
        KL_UCB_Standard, tc.probs, tc.horizon, num_sims, base_seed)
    optimized_regret, optimized_time, optimized_trajectories = simulate_with_trajectories(
        KL_UCB_Optimized, tc.probs, tc.horizon, num_sims, base_seed)
    
    speedup = standard_time / optimized_time if optimized_time > 0 else 0
    
    trajectories_identical = True
    max_trajectory_diff = 0.0
    trajectory_differences = []
    
    for i in range(num_sims):
        std_traj = standard_trajectories[i]
        opt_traj = optimized_trajectories[i]
        
        for t in range(len(std_traj)):
            diff = abs(std_traj[t] - opt_traj[t])
            trajectory_differences.append(diff)
            max_trajectory_diff = max(max_trajectory_diff, diff)
            
            if diff > 1e-6:
                trajectories_identical = False
    
    avg_trajectory_diff = sum(trajectory_differences) / len(trajectory_differences) if trajectory_differences else 0.0
    
    regret_diff = abs(standard_regret - optimized_regret)
    regret_tolerance = 1e-6
    regret_similar = regret_diff <= regret_tolerance
    
    regret_acceptable = optimized_regret <= tc.regret_threshold
    
    speedup_acceptable = speedup >= tc.speedup_threshold
    
    score = 0
    score += regret_acceptable
    score += speedup_acceptable

    return {
        'score': score,
        'standard_regret': standard_regret,
        'standard_time': standard_time,
        'optimized_regret': optimized_regret,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'regret_threshold': tc.regret_threshold,
        'speedup_threshold': tc.speedup_threshold,
        'regret_acceptable': regret_acceptable,
        'speedup_acceptable': speedup_acceptable,
        'regret_similar': regret_similar,
        'trajectories_identical': trajectories_identical,
        'regret_diff': regret_diff,
        'regret_tolerance': regret_tolerance,
        'max_trajectory_diff': max_trajectory_diff,
        'avg_trajectory_diff': avg_trajectory_diff,
        'num_trajectory_points': len(trajectory_differences)
    }

def grade_task3_bonus(tc_path):
    """Grade the bonus KL-UCB algorithm with 3 runs and take the best score."""
    best_result = None
    best_score = -1
    
    print("  Running 3 evaluations and taking the best result...")
    
    for run_num in range(3):
        print(f"    Run {run_num + 1}/3...", end=" ")
        result = _grade_task3_bonus_single_run(tc_path)
        print(f"Score: {result['score']}")
        
        if result['score'] > best_score:
            best_score = result['score']
            best_result = result
    
    print(f"  Best score across 3 runs: {best_score}")
    return best_result

def _grade_task3_bonus_single_run(tc_path):
    
    tc = read_tc(tc_path)
    
    num_sims = 3
    base_seed = 42
    
    def simulate_with_trajectories_bonus(algorithm_class, probs, horizon, num_sims, base_seed):
        """Simulate and return both average metrics and individual trajectories."""
        def single_sim_with_trajectory(seed, algorithm_class, probs, horizon):
            """Single simulation returning complete regret trajectory."""
            np.random.seed(seed)
            shuffled_probs = np.random.permutation(probs)
            bandit = BernoulliBandit(probs=shuffled_probs)
            
            start_time = time.time()
            algo_inst = algorithm_class(num_arms=len(shuffled_probs), horizon=horizon)
            
            regret_trajectory = []
            for t in range(horizon):
                arm_to_be_pulled = algo_inst.give_pull()
                reward = bandit.pull(arm_to_be_pulled)
                algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
                regret_trajectory.append(bandit.regret())
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            return regret_trajectory, execution_time
        
        all_trajectories = []
        all_times = []
        
        for i in range(num_sims):
            seed = base_seed + i
            trajectory, exec_time = single_sim_with_trajectory(seed, algorithm_class, probs, horizon)
            all_trajectories.append(trajectory)
            all_times.append(exec_time)
        
        avg_final_regret = sum(traj[-1] for traj in all_trajectories) / num_sims
        avg_time = sum(all_times) / num_sims
        
        return avg_final_regret, avg_time, all_trajectories
    
    standard_regret, standard_time, standard_trajectories = simulate_with_trajectories_bonus(
        KL_UCB_Standard, tc.probs, tc.horizon, num_sims, base_seed)
    bonus_regret, bonus_time, bonus_trajectories = simulate_with_trajectories_bonus(
        KL_UCB_Bonus, tc.probs, tc.horizon, num_sims, base_seed)
    
    speedup = standard_time / bonus_time if bonus_time > 0 else 0
    
    trajectories_identical = True
    max_trajectory_diff = 0.0
    trajectory_differences = []
    
    for i in range(num_sims):
        std_traj = standard_trajectories[i]
        bonus_traj = bonus_trajectories[i]
        
        for t in range(len(std_traj)):
            diff = abs(std_traj[t] - bonus_traj[t])
            trajectory_differences.append(diff)
            max_trajectory_diff = max(max_trajectory_diff, diff)
            
            if diff > 1e-6:
                trajectories_identical = False
    
    avg_trajectory_diff = sum(trajectory_differences) / len(trajectory_differences) if trajectory_differences else 0.0
    
    regret_acceptable = bonus_regret <= tc.regret_threshold
    
    speedup_acceptable = speedup >= tc.speedup_threshold
    
    score = 1 if (regret_acceptable and speedup_acceptable and trajectories_identical) else 0
    
    return {
        'score': score,
        'standard_regret': standard_regret,
        'standard_time': standard_time,
        'bonus_regret': bonus_regret,
        'bonus_time': bonus_time,
        'speedup': speedup,
        'regret_threshold': tc.regret_threshold,
        'speedup_threshold': tc.speedup_threshold,
        'regret_acceptable': regret_acceptable,
        'speedup_acceptable': speedup_acceptable,
        'trajectories_identical': trajectories_identical,
        'max_trajectory_diff': max_trajectory_diff,
        'avg_trajectory_diff': avg_trajectory_diff,
        'num_trajectory_points': len(trajectory_differences),
        'strict_tolerance': 1e-10
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='The task to run. Valid values are: 1, 2, 3, bonus, all')
    parser.add_argument('--algo', type=str, required=False, help='The algo to run (for task 1 only). Valid values are: ucb, kl_ucb, thompson, all')
    args = parser.parse_args()
    pass_fail = ['FAILED', 'PASSED']

    start = time.time()
    if args.task == '1' or args.task == 'all':
        if args.task == 'all':
            args.algo = 'all'
        if args.algo is None:
            print('Please specify an algorithm for task 1')
            exit(1)
        if args.algo.lower() not in ['ucb', 'kl_ucb', 'thompson', 'all']:
            print('Invalid algorithm')
            exit(1)

        print("="*18+" Task 1 "+"="*18)
        for i in range(1, 4):
            print(f"Testcase {i}")
            scores, regrets = grade_task1(f'testcases/task1-{i}.txt', args.algo)
            for algo, score in scores.items():
                print("{:18}: {}. Regret: {:.2f}".format(algo, pass_fail[score], regrets[algo]))
            print("")

    if args.task == '2' or args.task == 'all':
        print("="*18+" Task 2 "+"="*18)
        passed = grade_task2()

        total_testcases = 8   # number of public testcases
        score = (passed / total_testcases) * 1.5
        print(f"\nTask 2 Score: {score:.2f}/1.50 marks "
              f"({passed}/{total_testcases} public testcases passed)")
        print("="*50)
    
    if args.task == '3' or args.task == 'all':
        print("="*18+" Task 3 "+"="*18)
        task3_files = [f for f in os.listdir('testcases') if f.startswith('task3-')]
        if not task3_files:
            print("No task 3 test cases found.")
        else:
            total_marks = 0.0
            max_marks_per_testcase = 2.0  # Total marks per testcase (excluding 1 mark for report)
            
            for i in range(1, len(task3_files) + 1):
                tc_file = f'testcases/task3-{i}.txt'
                if os.path.exists(tc_file):
                    print(f"Testcase {i}")
                    testcase_marks = 0.0
                    
                    # Test Optimized KL-UCB
                    result = grade_task3(tc_file)
                    print("Optimized KL-UCB Algorithm: {}".format(pass_fail[1 if result['score'] > 0 else 0]))
                    print("  Standard  - Regret: {:.2f}, Time: {:.4f}s".format(
                        result['standard_regret'], result['standard_time']))
                    print("  Optimized - Regret: {:.2f}, Time: {:.4f}s".format(
                        result['optimized_regret'], result['optimized_time']))
                    print("  Speedup: {:.2f}x (Required: {:.2f}x) - {}".format(
                        result['speedup'], result['speedup_threshold'], 
                        "PASS" if result['speedup_acceptable'] else "FAIL"))
                    print("  Regret within threshold: {} (Max: {:.2f})".format(
                        "PASS" if result['regret_acceptable'] else "FAIL", result['regret_threshold']))
                    
                    # Calculate marks for Optimized KL-UCB (use the actual score from best run)
                    optimized_marks = float(result['score'])
                    
                    # Display individual criteria results for transparency
                    if result['regret_acceptable']:
                        print("  Regret Threshold Mark: 1.0/1.0")
                    else:
                        print("  Regret Threshold Mark: 0.0/1.0")
                    
                    if result['speedup_acceptable']:
                        print("  Speedup Threshold Mark: 1.0/1.0")
                    else:
                        print("  Speedup Threshold Mark: 0.0/1.0")
                    
                    if (result['trajectories_identical']) :
                        print("  Trajectories identical : PASS")
                    print("  Optimized KL-UCB Total: {:.1f}/2.0 marks".format(optimized_marks))
                    testcase_marks += optimized_marks
                    
                    print("  Testcase {i} Total: {:.1f}/2.0 marks".format(testcase_marks, i=i))
                    print("")
                    
                    total_marks += testcase_marks
            
            # Calculate normalized score
            num_testcases = len(task3_files)
            normalized_score = total_marks / num_testcases if num_testcases > 0 else 0.0
            
            print("="*50)
            print("TASK 3 SUMMARY:")
            print("Total marks across all testcases: {:.1f}/{:.1f}".format(total_marks, 2.0 * num_testcases))
            print("Number of testcases: {}".format(num_testcases))
            print("Normalized score: {:.2f}/2.00 marks (excluding 2 mark for report)".format(total_marks / num_testcases))
            print("="*50)

    if args.task == 'bonus':
        print("="*15+" BONUS TESTING "+"="*15)
        bonus_files = [f for f in os.listdir('testcases') if f.startswith('bonus-')]
        if not bonus_files:
            print("No bonus test cases found.")
        else:
            passed_testcases = 0
            total_testcases = 0
            all_results = []
            
            for i in range(1, len(bonus_files) + 1):
                bonus_file = f'testcases/bonus-{i}.txt'
                if os.path.exists(bonus_file):
                    total_testcases += 1
                    print(f"Bonus Testcase {i}")
                    
                    # Test Bonus KL-UCB
                    result = grade_task3_bonus(bonus_file)
                    all_results.append(result)
                    
                    if 'error' in result:
                        print("Bonus KL-UCB Algorithm: NOT IMPLEMENTED")
                        print(f"  Error: {result['error']}")
                        print("  Bonus Testcase {i}: FAILED")
                    else:
                        print("Bonus KL-UCB Algorithm: {}".format(pass_fail[result['score']]))
                        print("  Standard - Regret: {:.2f}, Time: {:.4f}s".format(
                            result['standard_regret'], result['standard_time']))
                        print("  Bonus    - Regret: {:.2f}, Time: {:.4f}s".format(
                            result['bonus_regret'], result['bonus_time']))
                        print("  Speedup: {:.2f}x (Required: {:.2f}x) - {}".format(
                            result['speedup'], result['speedup_threshold'], 
                            "PASS" if result['speedup_acceptable'] else "FAIL"))
                        print("  Regret within threshold: {} (Max: {:.2f})".format(
                            "PASS" if result['regret_acceptable'] else "FAIL", result['regret_threshold']))
                        print("  Trajectories identical: {} (Max diff: {:.2e}, Tolerance: {:.2e})".format(
                            "PASS" if result['trajectories_identical'] else "FAIL", 
                            result['max_trajectory_diff'], result['strict_tolerance']))
                        
                        if result['score'] == 1:
                            passed_testcases += 1
                            print("  Bonus Testcase {i}: PASSED".format(i=i))
                        else:
                            print("  Bonus Testcase {i}: FAILED".format(i=i))
                    
                    print("")
            
            # Calculate bonus marks: only award if ALL testcases pass
            bonus_marks_awarded = 1.0 if (passed_testcases == total_testcases and total_testcases > 0) else 0.0
            
            print("="*50)
            print("BONUS SUMMARY:")
            print("Testcases passed: {}/{} ".format(passed_testcases, total_testcases))
            print("Requirement: ALL testcases must pass to get bonus mark")
            if bonus_marks_awarded > 0:
                print("BONUS MARK AWARDED: 1.0/1.0 marks")
            else:
                print("BONUS MARK AWARDED: 0.0/1.0 marks")
            print("Total bonus marks: {:.1f}/1.0".format(bonus_marks_awarded))

    end = time.time()

    print("Time elapsed: {:.2f} seconds".format(end-start))
