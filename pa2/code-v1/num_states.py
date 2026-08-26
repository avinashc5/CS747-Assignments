from itertools import product

def nonterminal_states(threshold):
    total_sum = 2 * sum(range(1, 14))  # 182
    m = {}
    count = 0

    for deck in product([0, 1, 2], repeat=13):  # 3^13 possibilities
        deck_sum = sum((i + 1) * deck[i] for i in range(13))
        hand_sum = total_sum - deck_sum
        if hand_sum <= threshold:
            count += 1
            m[deck] = count

    return m, count

if __name__ == "__main__":
    m, result = nonterminal_states(26)
    print(result)
    print(m)
