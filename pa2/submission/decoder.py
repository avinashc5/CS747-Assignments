import argparse
import itertools

def all_cards():
    return [f"{n}H" for n in range(1, 14)] + [f"{n}D" for n in range(1, 14)]

def hand_sum(hand):
    return sum(int(card[:-1]) for card in hand)

def nonterminal_states(threshold):
    m = {}
    count = 0

    for deck in itertools.product([0, 1, 2], repeat=13):  # 3^13 possibilities
        hand_sum = sum((i + 1) * deck[i] for i in range(13))
        if hand_sum < threshold:
            m[deck] = count
            count += 1

    return m, count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--value_policy')
    parser.add_argument('--testcase')
    args = parser.parse_args()

    values = []
    policy = []
    with open(args.value_policy) as f:
        lines = [l.strip() for l in f]
        for line in lines:
            values.append(float(line.split()[0]))
            policy.append(int(line.split()[1]))

    with open(args.testcase) as f:
        lines = [l.strip() for l in f]
        threshold = int(lines[1])
        bonus = int(lines[2])
        bonus_seq = list(map(int, lines[3].split()))
        
        state_map, count = nonterminal_states(threshold)
        
        for line in lines[5:]:
            hand = line.split()
            state_list = 13 * [0]
            expanded_state = 26 * [0]
            for card in hand:
                expanded_state[int(card[:-1]) - 1 + (0 if card[-1] == 'H' else 13)] += 1
                state_list[int(card[:-1]) - 1] += 1
            state = tuple(state_list)
            action = policy[state_map[state]]
            out_action = 0
            if action == 0:
                out_action = 0
            elif 1 <= action <= 13:
                if expanded_state[action - 1] > 0:
                    out_action = action
                elif expanded_state[action - 1 + 13] > 0:
                    out_action = action + 13
                else:
                    out_action = 28 # Error
            elif action == 14:
                out_action = 27
            else:
                out_action = 28 # Error
            print(out_action)
            

if __name__ == "__main__":
    main()
