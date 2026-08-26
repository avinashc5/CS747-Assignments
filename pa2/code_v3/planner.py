import argparse
import numpy as np
import pulp

tol = 1e-8

def parse_mdp(path):
    with open(path) as f:
        lines = f.readlines()

    S = int(lines[0].split()[1])
    A = int(lines[1].split()[1])
    
    terminate_states = set(map(int, lines[2].split()[1:]))
    terminate_states = list(terminate_states)
    for t in terminate_states:
        if t < 0 or t >= S:
            terminate_states = []

    transitions = {s: {a: [] for a in range(A)} for s in range(S)}

    for line in lines:
        if not line.startswith("transition"):
            continue
        _, s1, a, s2, r, p = line.split()
        s1, a, s2 = int(s1), int(a), int(s2)
        r, p = float(r), float(p)
        transitions[s1][a].append((s2, r, p))

    mdptype = lines[-2].split()[1]
    gamma = float(lines[-1].split()[1])
    return S, A, transitions, terminate_states, mdptype, gamma


def parse_policy(policy_path, n_states):
    with open(policy_path, 'r') as f:
        policy = [int(line.strip()) for line in f.readlines()]
    assert len(policy) == n_states
    return policy

def improve_policy(S, A, V, transitions, gamma):
    new_policy = np.zeros(S, dtype=int)
    for s in range(S):
        q_values = np.zeros(A)
        for a in range(A):
            q_values[a] = sum(p * (r + gamma * V[s2]) for s2, r, p in transitions[s][a])
        new_policy[s] = np.argmax(q_values)
    return new_policy

def howard_policy_iteration(S, A, transitions, gamma, terminate_states):
    policy = np.zeros(S, dtype=int)
    for _ in range(50):
        V = evaluate_policy(policy, transitions, gamma, terminate_states)
        new_policy = improve_policy(S, A, V, transitions, gamma)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return V, policy

def evaluate_policy(policy, transitions, gamma, terminate_states):
    num_states = len(policy)
    V = np.zeros(num_states)

    for i in range(1000):
        V_new = np.zeros(num_states)
        for s in range(num_states):
            if s in terminate_states:
                V_new[s] = 0.0
                continue
            a = policy[s]
            total = 0.0
            for (s_next, r, p) in transitions[s][a]:
                total += p * (r + gamma * V[s_next])
            V_new[s] = total
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new

    return V

def linear_programming(n_states, n_actions, transitions, gamma, terminate_states):
    prob = pulp.LpProblem("MDP_LP", pulp.LpMinimize)

    V = {s: pulp.LpVariable(f"V_{s}", lowBound=None) for s in range(n_states)}

    prob += pulp.lpSum(V[s] for s in range(n_states))

    for s in range(n_states):
        for a in range(n_actions):
            rhs = pulp.lpSum(
                p * (r + gamma * V[s2])
                for s2, r, p in transitions[s][a]
            )
            prob += V[s] >= rhs
            
    if terminate_states:
        for t in terminate_states:
            prob += V[t] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    V_opt = np.array([pulp.value(V[s]) for s in range(n_states)])

    policy = []
    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            q = sum(p * (r + gamma * V_opt[s2]) for s2, r, p in transitions[s][a])
            q_values.append(q)
        policy.append(int(np.argmax(q_values)))

    return V_opt, policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', required=True, help='Path to MDP file')
    parser.add_argument('--algorithm', choices=['hpi', 'lp'], default='lp', help='Algorithm to use')
    parser.add_argument('--policy', help='Path to policy file')
    args = parser.parse_args()

    n_states, n_actions, transitions, terminate_states, mdptype, gamma = parse_mdp(args.mdp)
    if args.policy:
        policy = parse_policy(args.policy, n_states)
        V = evaluate_policy(policy, transitions, gamma, terminate_states)
        for s in range(n_states):
            print(f"{V[s]:.6f} {policy[s]}")
    else:
        if args.algorithm == 'lp':
            V, policy = linear_programming(n_states, n_actions, transitions, gamma, terminate_states)
        else:
            V, policy = howard_policy_iteration(n_states, n_actions, transitions, gamma, terminate_states)
        for s in range(n_states):
            print(f"{V[s]:.6f} {policy[s]}")

if __name__ == '__main__':
    main()