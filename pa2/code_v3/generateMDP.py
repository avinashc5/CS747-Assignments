#! /usr/bin/python
import random,argparse,sys
parser = argparse.ArgumentParser()
import numpy as np

class MDP():
    def __init__(self,S,A,gamma,mdptype,rseed):
        random.seed(rseed)
        if mdptype=="continuing":
            self.generateContinuingMDP(S,A,gamma,mdptype)
        else:
            assert mdptype=='episodic'
            self.generateEpisodicMDP(S,A,gamma,mdptype)
    
    def generateEpisodicMDP(self,S,A,gamma,mdptype):
        print("numStates",S)
        print("numActions",A)

        flag = 0
        while(flag == 0):
            if S>5:
                end_ls = random.sample(range(S), random.randint(0, max(2,S//10)))
            else:
                end_ls = random.sample(range(S), random.randint(0, min(2,S-2)))
            path = [i for i in range(S) if i not in end_ls]
            random.shuffle(path)
            end_ls.append(path[-1])
            end_print = ('end '+ ' '.join(map(str,end_ls)))
            trans_print_arr = []
            transitions = np.zeros((S, A, S), dtype=bool)
            #print(path)
            for s in range(0, S):
                for a in range(0, A):
                    if s not in end_ls:
                        degree = random.randint(1,min(5,S))
                        degree-=1
                        next_state_in_path = path[path.index(s)+1]
                        l = [i for i in range(S) if i!=next_state_in_path]
                        random.shuffle(l)
                        R = [random.uniform(-1,1) for i in range(degree+1)]
                        T = [random.randint(1,1000) for i in range(degree)]
                        if not len(T)==0:
                            T.append(random.randint(sum(T)//5,sum(T)))
                        else:
                            T.append(1)
                        sumT = sum(T)
                        for i in range(degree):
                            transitions[s][a][l[i]] = True
                            trans_print_arr.append("transition "+str(s) + ' ' + str(a) + ' '+ str(l[i]) + ' ' + str (R[i]) + ' ' + str(T[i]/sumT))
                        trans_print_arr.append("transition "+str(s) + ' ' + str(a) + ' '+ str(next_state_in_path) + ' ' + str (R[-1]) + ' ' + str(T[-1]/sumT))
                        transitions[s][a][next_state_in_path] = True
            #This portion checks if there is non-zero probability of termination for some action from all states
            flag = 1
            Yes_S = np.zeros(S,dtype=bool)
            for s in end_ls:
                Yes_S[s] = True
            good_list = end_ls
            old_count = 0
            new_count = sum(Yes_S)
            while(new_count > old_count):
                for s in range(S):
                    if(Yes_S[s] == False):
                        trans_flag = 1
                        for a in range(A):
                            act_flag = 0
                            for s2 in good_list:
                                if (transitions[s][a][s2]):
                                    act_flag = 1
                            if (act_flag == False):
                                trans_flag = 0
                                break
                        if(trans_flag):
                            good_list.append(s)
                            Yes_S[s] = True
                old_count = new_count
                new_count = sum(Yes_S)
            
            
            if (new_count != S):
                flag = 0


        print(end_print)
        for trans in trans_print_arr:
            print(trans)
        print("mdptype",mdptype)
        print("discount ",gamma)


    def generateContinuingMDP(self,S,A,gamma,mdptype):
        print("numStates",S)
        print("numActions",A)
        print("end -1")
        #start = random.randint(0, S-1)
        #print("start",start)
        #
        #end = random.sample(range(S), random.randint(1, S-1))
        #print("end",' '.join(map(str,end)))
        

        for s in range(0, S):
            for a in range(0, A):
                degree = random.randint(1,min(5,S))
                l = [i for i in range(S)]
                random.shuffle(l)
                R = [random.uniform(-1,1) for i in range(degree)]
                T = [random.random() for i in range(degree)]
                sumT = sum(T)
                for i in range(degree):
                    print("transition",s,a,l[i],R[i],T[i]/sumT)

        print("mdptype",mdptype)
        print("discount ",gamma)

if __name__ == "__main__":
    parser.add_argument("--S",type=int,default=5)
    parser.add_argument("--A",type=int,default=2)
    parser.add_argument("--gamma",type=float,default=0.9)
    parser.add_argument("--mdptype",type=str,default="continuing")
    parser.add_argument("--rseed",type=int,default=0)
    
    
    args = parser.parse_args()
    if not (args.S>1 and args.S<=100):
        print("number of states shoud be from 2 to 100")
        sys.exit(0)
    
    if not (args.A>1 and args.A<=100):
        print("number of actions shoud be from 2 to 100")
        sys.exit(0)
    
    if not (args.gamma>=0 and args.gamma<=1):
        print("gamma should be with in 0 to 1")
        sys.exit(0)
    if not (args.mdptype=="continuing" or args.mdptype=="episodic"):
        print("Type of MDP should be continuing or episodic")
        sys.exit(0)
    
    
    #print(args)
    algo = MDP(args.S,args.A,args.gamma,args.mdptype,args.rseed)






