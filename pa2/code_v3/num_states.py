import itertools
import pprint

def nonterminal_states(threshold):
    total_sum = 2 * sum(range(1, 14))  # 182
    m = {}
    count = 0

    for deck in itertools.product([0, 1, 2], repeat=13):  # 3^13 possibilities
        hand_sum = sum((i + 1) * deck[i] for i in range(13))
        if hand_sum < threshold:
            m[deck] = count
            count += 1

    return m, count

if __name__ == "__main__":
    m, result = nonterminal_states(int(input()))
    print(result)
    # pprint.pprint(m)
