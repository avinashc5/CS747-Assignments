#! /usr/bin/python3
from email import policy
import random,argparse,sys,subprocess,os
parser = argparse.ArgumentParser()
import numpy as np
random.seed(0)


input_file_ls = ['data/mdp/continuing-mdp-10-5.txt','data/mdp/continuing-mdp-2-2.txt', 'data/mdp/continuing-mdp-25-10.txt', 'data/mdp/continuing-mdp-50-20.txt','data/mdp/episodic-mdp-10-5.txt','data/mdp/episodic-mdp-2-2.txt', 'data/mdp/episodic-mdp-25-10.txt', 'data/mdp/episodic-mdp-50-20.txt']
flag_ok = 0


class VerifyOutputPlanner:
    def __init__(self,algorithm,print_error):
        algorithm_ls = list()
        if algorithm=='all':
            algorithm_ls+=['hpi','lp', 'default']
        else:
            algorithm_ls.append(algorithm)
            
        for algo in algorithm_ls:
            print('verify output',algo)
            counter = 1    
        
            for in_file in input_file_ls:
                print("\n\n","-"*100)
                if algo == 'default':
                    cmd_planner = "python3","planner.py","--mdp",in_file
                else:
                    cmd_planner = "python3","planner.py","--mdp",in_file,"--algorithm",algo
                print('test case',str(counter),algo,":\t"," ".join(cmd_planner))
                counter+=1
                cmd_output = subprocess.check_output(cmd_planner,universal_newlines=True)
                self.verifyOutput(cmd_output,in_file,print_error)
        policy_eval_files = ['data/mdp/continuing-mdp-10-5.txt', 'data/mdp/episodic-mdp-10-5.txt']
        for in_file in policy_eval_files:
            cmd_planner = "python3","planner.py","--mdp",in_file, "--policy", in_file.replace("continuing","policy-continuing").replace("episodic","policy-episodic")
            print('test case',str(counter),'policy evaluation',":\t"," ".join(cmd_planner))
            counter+=1
            cmd_output = subprocess.check_output(cmd_planner,universal_newlines=True)
            self.verifyOutput(cmd_output,in_file,print_error, pol_eval = True)
        
    def verifyOutput(self,cmd_output,in_file,pe, pol_eval = False):

        sol_file = in_file.replace("continuing","sol-continuing").replace("episodic","sol-episodic")
        if (pol_eval):
            sol_file = in_file.replace("continuing","sol-policy-continuing").replace("episodic","sol-policy-episodic")
        base = np.loadtxt(sol_file,delimiter=" ",dtype=float)
        output = cmd_output.split("\n")
        nstates = base.shape[0]
        
        est = [i.split() for i in output if i!='']
        
        
        mistakeFlag = False
        #Check1: Checking the number of lines printed
        if not len(est)==nstates:
            mistakeFlag = True
            print("\n","*"*10,"Mistake:Exact number of line in standard output should be",nstates,"but have",len(est),"*"*10)
            
        #Check2: Each line should have only two values
        for i in range(len(est)):
            if not len(est[i])==2:
                mistakeFlag = True
                print("\n","*"*10,"Mistake: On each line you should print only value,policy for a state","*"*10)
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
                return
            #calculating the error
            for i in range(len(est)):
                est_V = float(est[i][0]);base_V = float(base[i][0])
                print("%10.6f"%est_V,"%10.6f"%base_V,"%10.6f"%abs(est_V-base_V),end="\t")
                if abs(est_V-base_V) <= (10**-4):
                    print("OK")
                else:
                    flag_ok = 1
                    print("\tNot OK")
            

def run(gameconfig, test):
    cmd_encoder = "python3", "encoder.py", "--game_config", gameconfig
    print("\n","Generating the MDP encoding using encoder.py")
    f = open('verify_attt_mdp','w')
    subprocess.call(cmd_encoder,stdout=f)
    f.close()

    cmd_planner = "python3","planner.py","--mdp","verify_attt_mdp"
    print("\n","Generating the value policy file using planner.py using default algorithm")
    f = open('verify_attt_planner','w')
    subprocess.call(cmd_planner,stdout=f)
    f.close()

    cmd_decoder = "python3","decoder.py","--value_policy","verify_attt_planner","--testcase",test
    print("\n","Generating the decoded policy file using decoder.py")
    
    cmd_output = subprocess.check_output(cmd_decoder,universal_newlines=True)
        
    os.remove('verify_attt_mdp')
    os.remove('verify_attt_planner')
    return cmd_output


def verifyOutput(output, solution):
    outputs = [int(i) for i in output.split()]
    solutions = []

    with open(solution, 'r') as f:
        
        # format: Answer n1 n2 ...
        for line in f:
            s = line.split()[0:1]
            sol = [int(i) for i in s]
            solutions.append(sol)

    # check if output lies in sol
    flag_ok = 1
    for i in range(len(outputs)):
        if outputs[i] not in solutions[i]:
            flag_ok = 0
            print("Mistake: The output is not in the solution")
            print("Output: ", outputs[i])
            print("Solution: ", solutions[i])
        else:
            print("OK")

    if flag_ok:
        print("All checks passed")

if __name__ == "__main__":
    parser.add_argument('--task', type = int, default=1)
    parser.add_argument("--algorithm",type=str,default="default")
    parser.add_argument("--pe",type=str,default="yes")
    args = parser.parse_args()

    if(args.task == 1):
        algo = VerifyOutputPlanner(args.algorithm,args.pe)
        if(flag_ok):
            print("THERE IS A MISTAKE in Task 1")
            
    elif(args.task == 2):
        in_file_ls = ['data/gameconfig/gameconfig_' + str(i) + '.txt' for i in range(0, 5)]
        in_file_test_ls = ['data/test/test_' + str(i) + '.txt' for i in range(0, 5)]
        in_file_sol_ls = ['data/test/test_' + str(i) + '_solution.txt' for i in range(0, 5)]

        i = 0
        for in_file, in_file_test, in_file_sol in zip(in_file_ls, in_file_test_ls, in_file_sol_ls):
            print("Running for ", in_file) 
            output = run(in_file, in_file_test)
            verifyOutput(output, in_file_sol)
            i +=1
    else:
        algo = VerifyOutputPlanner(args.algorithm,args.pe)
        if(flag_ok):
            print("THERE IS A MISTAKE in Task 1")
        
        in_file_ls = ['data/gameconfig/gameconfig_' + str(i) + '.txt' for i in range(0, 5)]
        in_file_test_ls = ['data/test/test_' + str(i) + '.txt' for i in range(0, 5)]
        in_file_sol_ls = ['data/test/test_' + str(i) + '_solution.txt' for i in range(0, 5)]

        i = 0
        for in_file, in_file_test, in_file_sol in zip(in_file_ls, in_file_test_ls, in_file_sol_ls):
            print("Running for ", in_file) 
            output = run(in_file, in_file_test)
            verifyOutput(output, in_file_sol)
            i +=1

        

