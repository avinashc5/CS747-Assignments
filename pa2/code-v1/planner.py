import argparse
import numpy as np
import pulp

max_policy_eval_iter = 10000
tol = 1e-8

def parse_mdp(mdp_path):
    lines = []
    with open(mdp_path, 'r') as f:
        lines = f.readlines()
    n_states = int(lines[0].split()[1])
    n_actions = int(lines[1].split()[1])
    end = int(lines[2].split()[1])
    transitions = np.zeros((n_states, n_actions, n_states))
    rewards = np.zeros((n_states, n_actions, n_states))
    for line in lines[3:]:
        if line.startswith('transition'):
            parts = line.split()
            s1 = int(parts[1])
            ac = int(parts[2])
            s2 = int(parts[3])
            rewards[s1][ac][s2] = float(parts[4])
            prob = float(parts[5])
            transitions[s1][ac][s2] = prob
        elif line.startswith('mdptype'):
            mdp_type = line.split()[1]
        elif line.startswith('discount'):
            gamma = float(line.split()[1])
    return n_states, n_actions, rewards, transitions, gamma

def parse_policy(policy_path, n_states):
    with open(policy_path, 'r') as f:
        policy = [int(line.strip()) for line in f.readlines()]
    assert len(policy) == n_states
    return policy

def policy_evaluation(policy, rewards, transitions, gamma, tol=tol, max_iter=max_policy_eval_iter):
    n_states = len(policy)
    V = np.zeros(n_states)
    for _ in range(max_iter):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            a = policy[s]
            for s_next in range(n_states):
                V_new[s] += transitions[s][a][s_next] * (rewards[s][a][s_next] + gamma * V[s_next])
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V

def howards_policy_iteration(n_states, n_actions, rewards, transitions, gamma, tol=tol, max_iter=max_policy_eval_iter):
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)

    for _ in range(max_iter):
        V = policy_evaluation(policy, rewards, transitions, gamma, tol)
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s]
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for s_next in range(n_states):
                    action_values[a] += transitions[s][a][s_next] * (rewards[s][a][s_next] + gamma * V[s_next])
            best_action = np.argmax(action_values)
            policy[s] = best_action
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break
    return V, policy

def linear_programming(n_states, n_actions, rewards, transitions, gamma):
    
    # LP problem
    prob = pulp.LpProblem("MDP", pulp.LpMaximize)

    # Variables: V(s) for each state
    V = {s: pulp.LpVariable(f"V_{s}", lowBound=None) for s in range(n_states)}

    # Objective: maximize -sum_s V(s)
    prob += -pulp.lpSum([V[s] for s in range(n_states)])

    # Constraints: V(s) >= sum_{s'} T(s,a,s') * (R(s,a,s') + gamma*V(s'))
    for s in range(n_states):
        for a in range(n_actions):
            rhs = pulp.lpSum([
                transitions[s, a, sp] * (rewards[s][a][sp] + gamma * V[sp])
                for sp in range(n_states)
            ])
            prob += V[s] >= rhs

    # Solve LP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract solution
    V_opt = [pulp.value(V[s]) for s in range(n_states)]
    policy = [int(np.argmax([rewards[s][a] + gamma * np.dot(transitions[s][a], V_opt) for a in range(n_actions)])) for s in range(n_states)]
    return V_opt, policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', required=True, help='Path to MDP file')
    parser.add_argument('--algorithm', choices=['hpi', 'lp'], default='hpi', help='Algorithm to use')
    parser.add_argument('--policy', help='Path to policy file')
    args = parser.parse_args()

    n_states, n_actions, rewards, transitions, gamma = parse_mdp(args.mdp)

    if args.policy:
        policy = parse_policy(args.policy, n_states)
        V = policy_evaluation(policy, rewards, transitions, gamma)
        for s in range(n_states):
            print(f"{V[s]:.6f} {policy[s]}")
    else:
        if args.algorithm == 'lp':
            V, policy = linear_programming(n_states, n_actions, rewards, transitions, gamma)
        else:
            V, policy = howards_policy_iteration(n_states, n_actions,rewards, transitions, gamma)
        for s in range(n_states):
            print(f"{V[s]:.6f} {policy[s]}")

if __name__ == '__main__':
    main()