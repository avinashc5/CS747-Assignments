import argparse
import itertools

def hand_sum(hand) -> int:
    return sum((i + 1) * hand[i] for i in range(13))

def bonus_seq_in_hand(state, bonus_seq) -> bool:
    return state[bonus_seq[0] - 1] > 0 and state[bonus_seq[1] - 1] > 0 and state[bonus_seq[2] - 1] > 0
                
def nonterminal_states(threshold):
    total_sum = 2 * sum(range(1, 14))  # 182
    m = {}
    count = 0

    for deck in itertools.product([0, 1, 2], repeat=13):  # 3^13 possibilities
        hand_sum = sum((i + 1) * deck[i] for i in range(13))
        if hand_sum <= threshold:
            m[deck] = count
            count += 1

    return m, count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_config", required=True)
    args = parser.parse_args()

    with open(args.game_config) as f:
        lines = [l.strip() for l in f]
        threshold = int(lines[1])
        bonus = int(lines[2])
        bonus_seq = list(map(int, lines[3].split()))
    
    state_map, count = nonterminal_states(threshold)
    
    states = list(state_map.keys())
    print(f"numStates {count + 1}")
    
    transitions = []
    
    # action 0 is for add
    for state in states:
        sum_of_hand = hand_sum(state)
        next_card_prob = 13*[0]
        remaining_cards = 26 - sum(state)
        for card in range(13):
            next_card_prob[card] = (2 - state[card])/remaining_cards
            terminate_prob = 0
            if state[card] != 2:
                if sum_of_hand + card + 1 <= threshold:
                    next_state_list = list(state)
                    next_state_list[card] = next_state_list[card] + 1
                    next_state = tuple(next_state_list)
                    # reward = card
                    # if not bonus_seq_in_hand(state, bonus_seq) and bonus_seq_in_hand(next_state, bonus_seq):
                    #     reward += bonus
                    reward = 0
                    transitions.append([state_map[state], 0, state_map[next_state], reward, next_card_prob[card]])
                else:
                    terminate_prob += next_card_prob[card]
        if terminate_prob > 0:
            # reward = -1 * sum_of_hand
            # if bonus_seq_in_hand(state, bonus_seq):
            #     reward -= bonus
            reward = 0
            transitions.append([state_map[state], 0, count, reward, terminate_prob])
    
    # action 1-13 is for swap(1) to swap(13)
    for action in range(1, 14):
        swap_card_idx = action - 1
        for state in states:
            if state[swap_card_idx] == 0:
                continue
            sum_of_hand = hand_sum(state)
            next_card_prob = 13*[0]
            remaining_cards = 26 - sum(state)
            for card in range(13):
                next_card_prob[card] = (2 - state[card])/remaining_cards
                terminate_prob = 0
                if state[card] != 2:
                    if sum_of_hand + card + 1 - action <= threshold:
                        next_state_list = list(state)
                        next_state_list[card] += 1
                        next_state_list[swap_card_idx] -= 1
                        next_state = tuple(next_state_list)
                        # reward = card
                        # if (not bonus_seq_in_hand(state, bonus_seq)) and bonus_seq_in_hand(next_state, bonus_seq):
                        #     reward += bonus
                        reward = 0
                        transitions.append([state_map[state], action, state_map[next_state], reward, next_card_prob[card]])
                    else:
                        terminate_prob += next_card_prob[card]
            if terminate_prob > 0:
                # reward = -1 * sum_of_hand
                # if bonus_seq_in_hand(state, bonus_seq):
                #     reward -= bonus
                reward = 0
                transitions.append([state_map[state], action, count, reward, terminate_prob])

    # action 14 is for stop
    for state in states:
        if bonus_seq_in_hand(state, bonus_seq):
            transitions.append([state_map[state], 14, count, sum_of_hand + bonus, 1])
        else:
            transitions.append([state_map[state], 14, count, sum_of_hand, 1])
    
    
    print("numActions 15")
    print("end " + str(count))
    for transition in transitions:
        print(f'transition {transition[0]} {transition[1]} {transition[2]} {transition[3]} {transition[4]}')
    print("mdptype episodic")
    print("discount 1.0")

if __name__ == "__main__":
    main()
