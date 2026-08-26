#! /usr/bin/python3
from email import policy
import shlex
import random, argparse, sys, subprocess, os
import numpy as np
import re
import time
import datetime

status_path = "status.txt"
parser = argparse.ArgumentParser()
random.seed(0)

input_file_ls = [
    'data/mdp/continuing-mdp-10-5.txt',
    'data/mdp/continuing-mdp-2-2.txt',
    'data/mdp/continuing-mdp-25-10.txt',
    'data/mdp/continuing-mdp-50-20.txt',
    'data/mdp/episodic-mdp-10-5.txt',
    'data/mdp/episodic-mdp-2-2.txt',
    'data/mdp/episodic-mdp-25-10.txt',
    'data/mdp/episodic-mdp-50-20.txt'
]
private_input_file_ls = [
    "private/mdp/mdp10000506c.txt",
    "private/mdp/mdp2505008c.txt",
    "private/mdp/mdp1002501c.txt",
    "private/mdp/mdp502002c.txt",
    "private/mdp/mdp10000310e.txt",
    "private/mdp/mdp2503004e.txt",
    "private/mdp/mdp1002508e.txt",
    "private/mdp/mdp502005e.txt"
]
flag_ok = 0


import subprocess
import numpy as np

class VerifyOutputPlanner:
    def __init__(self, algorithm, print_error, planner_path):
        print("\n\n", "="*100)
        print("TASK 1")
        print( "="*100)
        print("Verifying planner.py using", algorithm, "algorithm")
        print('-'*100)
        algorithm_ls = []
        if algorithm == 'all':
            algorithm_ls += ['hpi', 'lp', 'default']
        else:
            algorithm_ls.append(algorithm)

        self.public_testcase_scores = []   # store marks per testcase
        self.private_testcase_scores = []
        self.public_total_score = 0.0      # cumulative marks
        self.private_total_score = 0.0

        for algo in algorithm_ls:
            print('verify output', algo)
            counter = 1    

            for in_file in input_file_ls:
                print("\n\n", "-" * 100)
                if algo == 'default':
                    cmd_planner = ["python3", planner_path, "--mdp", in_file]
                else:
                    cmd_planner = ["python3", planner_path, "--mdp", in_file, "--algorithm", algo]

                print('test case', str(counter), algo, ":\t", " ".join(cmd_planner))
                counter += 1
                try:
                    cmd_output = subprocess.check_output(cmd_planner, universal_newlines=True, timeout=120)
                    passed = self.verifyOutput(cmd_output, in_file, print_error)
                except subprocess.TimeoutExpired as e:
                    print(f"Timeout expired on {in_file}")
                    passed = 0
                except subprocess.CalledProcessError as e:
                    print(f"Runtime error on {in_file}: {e}")
                    passed = 0
                
                # ✅ Store score
                if passed == 1:
                    self.public_testcase_scores.append(0.25)
                else:
                    self.public_testcase_scores.append(0)

            for in_file in private_input_file_ls:
                print("\n\n", "-" * 100)
                if algo == 'default':
                    cmd_planner = ["python3", planner_path, "--mdp", in_file]
                else:
                    cmd_planner = ["python3", planner_path, "--mdp", in_file, "--algorithm", algo]

                print('test case', str(counter), algo, ":\t", " ".join(cmd_planner))
                counter += 1
                try:
                    cmd_output = subprocess.check_output(cmd_planner, universal_newlines=True, timeout=120)
                    passed = self.verifyOutput(cmd_output, in_file, print_error)
                except subprocess.TimeoutExpired as e:
                    print(f"Timeout expired on {in_file}")
                    passed = 0
                except subprocess.CalledProcessError as e:
                    print(f"Runtime error on {in_file}: {e}")
                    passed = 0
                
                # ✅ Store score
                if passed == 1:
                    self.private_testcase_scores.append(0.25)
                else:
                    self.private_testcase_scores.append(0)

            
            # ---- Policy evaluation ----
            policy_eval_files = [
                'data/mdp/continuing-mdp-10-5.txt',
                'data/mdp/episodic-mdp-10-5.txt'
            ]
            for in_file in policy_eval_files:
                cmd_planner = [
                    "python3", planner_path,
                    "--mdp", in_file,
                    "--policy", in_file.replace("continuing", "policy-continuing")
                                      .replace("episodic", "policy-episodic")
                ]
                print('test case', str(counter), 'policy evaluation', ":\t", " ".join(cmd_planner))
                counter += 1
                try:
                    cmd_output = subprocess.check_output(cmd_planner, universal_newlines=True, timeout=120)
                    passed = self.verifyOutput(cmd_output, in_file, print_error, pol_eval=True)
                except subprocess.TimeoutExpired as e:
                    print(f"Timeout expired on {in_file}")
                    passed = 0
                except subprocess.CalledProcessError as e:
                    print(f"Runtime error on policy eval: {e}")
                    passed = 0

                if passed == 1:
                    self.public_testcase_scores.append(0.25)
                else:
                    self.public_testcase_scores.append(0)
            # ---- Private policy evaluation ----
            private_policy_eval_files = [
                "private/mdp/mdp502002c.txt",
                "private/mdp/mdp502005e.txt"
            ]
            privatepolicyfiles = [
                "private/mdp/policy-continuing-mdp-50-20.txt",
                "private/mdp/policy-episodic-mdp-50-20.txt"
            ]
            for in_file,pol_file in zip(private_policy_eval_files, privatepolicyfiles):
                cmd_planner = [
                    "python3", planner_path,
                    "--mdp", in_file,
                    "--policy", pol_file
                ]
                print('test case', str(counter), 'policy evaluation', ":\t", " ".join(cmd_planner))
                counter += 1
                try:
                    cmd_output = subprocess.check_output(cmd_planner, universal_newlines=True, timeout=120)
                    passed = self.verifyOutput(cmd_output, pol_file[0:12]+pol_file[19:], print_error, pol_eval=True)
                except subprocess.TimeoutExpired as e:
                    print(f"Timeout expired on {in_file}")
                    passed = 0
                except subprocess.CalledProcessError as e:
                    print(f"Runtime error on policy eval: {e}")
                    passed = 0

                if passed == 1:
                    self.private_testcase_scores.append(0.25)
                else:
                    self.private_testcase_scores.append(0)

        # ---- Print total ----
        self.public_total_score = sum(self.public_testcase_scores)
        self.private_total_score = sum(self.private_testcase_scores)
        print("\nFinal per-testcase Public scores:", self.public_testcase_scores)
        print(f"Total Public marks: {self.public_total_score:.3f}")
        print("\nFinal per-testcase Private scores:", self.private_testcase_scores)
        print(f"Total Private marks: {self.private_total_score:.3f}")

        print("\n\n", "="*100)
        print("TASK 1 COMPLETED")
        print("="*100)

    def verifyOutput(self, cmd_output, in_file, pe, pol_eval=False):
        sol_file = None
       
        if re.search(r"/mdp\d", in_file):  # only match files like .../mdp123...
            sol_file = re.sub(r"/(mdp)(\d)", r"/sol-\1\2", in_file)
        else:
            sol_file = in_file.replace("continuing", "sol-continuing").replace("episodic", "sol-episodic")
        if pol_eval:
            sol_file = in_file.replace("continuing", "sol-policy-continuing").replace("episodic", "sol-policy-episodic")
        print("Sol file:", sol_file)
        base = np.loadtxt(sol_file, delimiter=" ", dtype=float)
        output = cmd_output.split("\n")
        nstates = base.shape[0]
        est = [i.split() for i in output if i != '']
        
        mistakeFlag = False
        if not len(est) == nstates:
            mistakeFlag = True
            print("\n", "*"*10, f"Mistake: Exact number of lines should be {nstates}, but got {len(est)}", "*"*10)
            
        for i in range(len(est)):
            if not len(est[i]) == 2:
                mistakeFlag = True
                print("\n", "*"*10, "Mistake: Each line should have only two values", "*"*10)
                break
        
        if not mistakeFlag:
            print("ALL CHECKS PASSED!")
        else:
            print("You haven't printed output in correct format.")
            
        pe_ls = ['no','NO','No','nO']
        if pe not in pe_ls:
            if not mistakeFlag:
                print("Calculating error of your value function...")
            else:
                print("\nExiting without calculating error of your value function")
                return 0

            flag_ok = 0
            for i in range(len(est)):
                est_V = float(est[i][0]); base_V = float(base[i][0])
                print("%10.6f" % est_V, "%10.6f" % base_V, "%10.6f" % abs(est_V - base_V), end="\t")
                if abs(est_V - base_V) <= (10 ** -4):
                    print("OK")
                else:
                    flag_ok = 1
                    print("\tNot OK")

            if flag_ok == 0:
                print("✅ Test Passed")
                return 1
            else:
                print("❌ Test Failed")
                return 0
        else:
            # if print_error is 'no', only structural checks apply
            return 0 if mistakeFlag else 1


def run(gameconfig, test, planner_path, encoder_path, decoder_path):
    print("\nRunning full pipeline...")

    # --- Use shlex.quote() to safely quote each path ---
    encoder_path_q = shlex.quote(encoder_path)
    planner_path_q = shlex.quote(planner_path)
    decoder_path_q = shlex.quote(decoder_path)
    gameconfig_q = shlex.quote(gameconfig)
    test_q = shlex.quote(test)

    cmd_string = (
        f"python3 -u {encoder_path_q} --game_config {gameconfig_q} > verify_attt_mdp && "
        f"python3 -u {planner_path_q} --mdp verify_attt_mdp > verify_attt_planner && "
        f"python3 -u {decoder_path_q} --value_policy verify_attt_planner --testcase {test_q}"
    )
    # Execute the command string using a shell
    # MODIFIED: Removed the extra 'cmd_output =' line that was duplicated
    cmd_output = subprocess.run(cmd_string, shell=True, universal_newlines=True, timeout=120, capture_output=True)
    os.remove('verify_attt_mdp')
    os.remove('verify_attt_planner')
    return cmd_output.stdout


def verifyOutput(output, solution):
    outputs = [int(i) for i in output.split()]
    solutions = []

    with open(solution, 'r') as f:
        for line in f:
            s = line.split()
            sol = [int(i) for i in s]
            solutions.append(sol)
    print(outputs, solutions)
    flag_ok = 1
    if len(outputs) != len(solutions):
        flag_ok = 0
        print("Mistake: Number of lines in decoder's output and solution file are different")
    else: 
        for i in range(len(outputs)):
            if outputs[i] not in solutions[i]:
                flag_ok = 0
                print("Mistake: The output is not in the solution")
                print("Output:", outputs[i])
                print("Solution:", solutions[i])
            else:
                print("OK")

    if flag_ok:
        print("All checks passed")

    return flag_ok


def run_task2(planner_path, encoder_path, decoder_path):
    print("\n\n", "=" * 100)
    print("TASK 2")
    print("=" * 100)

    in_file_ls = [f"data/gameconfig/gameconfig_{i}.txt" for i in range(0, 5)]
    in_file_test_ls = [f"data/test/test_{i}.txt" for i in range(0, 5)]
    in_file_sol_ls = [f"data/test/test_{i}_solution.txt" for i in range(0, 5)]

    private_file_ls = [f"private/gameconfig/gameconfig_{i}.txt" for i in range(0, 5)]
    private_file_test_ls = [f"private/test/test_{i}.txt" for i in range(0, 5)]
    private_file_sol_ls = [f"private/test/test_{i}_solution.txt" for i in range(0, 5)]

    public_testcase_scores = []
    private_testcase_scores = []
    weight_per_test = 0.5  # marks per passing test

    for in_file, in_file_test, in_file_sol in zip(in_file_ls, in_file_test_ls, in_file_sol_ls):
        print("\nRunning for", in_file)
        try:
            output = run(in_file, in_file_test, planner_path, encoder_path, decoder_path)
            ok = verifyOutput(output, in_file_sol)
            if ok:
                public_testcase_scores.append(weight_per_test)
            else:
                public_testcase_scores.append(0)
        except Exception as e:
            print(f"Error running for {in_file}: {e}")
            public_testcase_scores.append(0)
    for in_file, in_file_test, in_file_sol in zip(private_file_ls, private_file_test_ls, private_file_sol_ls):
        print("\nRunning for", in_file)
        try:
            output = run(in_file, in_file_test, planner_path, encoder_path, decoder_path)
            ok = verifyOutput(output, in_file_sol)
            if ok:
                private_testcase_scores.append(weight_per_test)
            else:
                private_testcase_scores.append(0)
        except Exception as e:
            print(f"Error running for {in_file}: {e}")
            private_testcase_scores.append(0)

    public_total_score = sum(public_testcase_scores)
    private_total_score = sum(private_testcase_scores)
    print("\nFinal per-testcase Public scores:", public_testcase_scores)
    print("Total Public marks:", public_total_score)
    print("\nFinal per-testcase Private scores:", private_testcase_scores)
    print("Total Private marks:", private_total_score)
    print("\n\n", "=" * 100)
    print("TASK 2 COMPLETED")
    print("=" * 100)

if __name__ == "__main__":
    parser.add_argument('--task', type=int, default=None)
    parser.add_argument("--algorithm", type=str, default="default")
    parser.add_argument("--pe", type=str, default="yes")

    # ✅ NEW ARGUMENTS
    parser.add_argument("--planner", type=str, required=True, help="Path to planner.py")
    parser.add_argument("--encoder", type=str, default=None, help="Path to encoder.py (for task 2)")
    parser.add_argument("--decoder", type=str, default=None, help="Path to decoder.py (for task 2)")

    args = parser.parse_args()

    if args.task == 1:
        algo = VerifyOutputPlanner(args.algorithm, args.pe, args.planner)
        if flag_ok:
            print("THERE IS A MISTAKE in Task 1")
            
    elif args.task == 2:
        run_task2(args.planner, args.encoder, args.decoder)
    else:
        algo = VerifyOutputPlanner(args.algorithm, args.pe, args.planner)
        if flag_ok:
            print("THERE IS A MISTAKE in Task 1")

        run_task2(args.planner, args.encoder, args.decoder)
