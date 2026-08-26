import argparse
import tkinter as tk
import random
import sys

# --- Global Variables ---
all_suits = ['♥', '♦']
deck = []
agent_hand = []
selected_index = None
selected_rect = None
max_sum = 0
bonus = 0
special_seq = []

CARD_WIDTH = 40
CARD_HEIGHT = 60

# --- Game Logic Functions (unchanged) ---
def hand_sum():
    return sum(v for v, _ in agent_hand)

def final_score():
    total = hand_sum()
    if total > max_sum:
        return 0
    hand_values = [v for v, _ in agent_hand]
    if all(val in hand_values for val in special_seq):
        total += bonus
    return total

# --- GUI Functions ---
def update_sums():
    agent_total = hand_sum()
    agent_sum_label.config(text=f"Agent's Sum: {agent_total}")
    max_sum_label.config(text=f"Max Sum: {max_sum}")
    special_seq_label.config(text=f"Special Sequence: {special_seq}")
    if agent_total > max_sum:
        stop_game(reason="Sum exceeded max!")

def draw_card(canvas, x, y, card, idx=None):
    v, s = card
    tag = f"card_{idx}" if idx is not None else None
    rect = canvas.create_rectangle(
        x, y, x + CARD_WIDTH, y + CARD_HEIGHT,
        fill="white", outline="black", width=2, tags=tag
    )
    canvas.create_text(x + 8, y + 8, text=str(v), fill="red", anchor="nw", font=("Arial", 10, "bold"), tags=tag)
    canvas.create_text(x + CARD_WIDTH / 2, y + CARD_HEIGHT / 2, text=s, fill="red", font=("Arial", 18), tags=tag)
    canvas.create_text(x + CARD_WIDTH - 5, y + CARD_HEIGHT - 5, text=str(v), fill="red", anchor="se", font=("Arial", 10, "bold"), tags=tag)
    if idx is not None:
        canvas.tag_bind(tag, "<Button-1>", lambda event, i=idx, r=rect: select_card(i, r))

def update_display():
    deck_canvas.delete("all")
    hand_canvas.delete("all")
    max_cols = 13
    x_start, y_start = 10, 10
    x, y = x_start, y_start
    for i, c in enumerate(deck):
        draw_card(deck_canvas, x, y, c)
        x += CARD_WIDTH + 5
        if (i + 1) % max_cols == 0:
            x, y = x_start, y + CARD_HEIGHT + 5
    x, y = x_start, y_start
    for i, c in enumerate(agent_hand):
        draw_card(hand_canvas, x, y, c, idx=i)
        x += CARD_WIDTH + 5
        if (i + 1) % max_cols == 0:
            x, y = x_start, y + CARD_HEIGHT + 5
    update_sums()

def pull_card():
    global selected_index, selected_rect
    if deck:
        selected_index = None
        selected_rect = None
        random.shuffle(deck)
        card = deck.pop(0)
        agent_hand.append(card)
        update_display()

def select_card(i, rect):
    global selected_index, selected_rect
    if selected_rect is not None:
        hand_canvas.itemconfig(selected_rect, outline="black", width=2)
    selected_index, selected_rect = i, rect
    hand_canvas.itemconfig(rect, outline="blue", width=3)

def swap_card():
    global selected_index
    if deck and agent_hand and selected_index is not None:
        random.shuffle(deck)
        new_card = deck.pop(0)
        old_card = agent_hand[selected_index]
        agent_hand[selected_index] = new_card
        deck.append(old_card)
        selected_index = None
        update_display()
    else:
        if args.policy is None:
            result_label.config(text="Select a card in your hand to swap!")

# --- The stop_game function to show the centered message ---
def stop_game(reason="Game stopped."):
    result_label.config(text="")

    score = final_score()
    if score == 0 and hand_sum() > max_sum:
        final_message = f"Busted!\nSum: {hand_sum()}\nFinal Score: 0"
    else:
        final_message = f"Game Over!\nFinal Score: {score}"
    
    game_over_label.config(text=final_message)
    game_over_label.place(relx=0.5, rely=0.3, anchor="center")
    
    for child in button_frame.winfo_children():
        child.config(state="disabled")

# --- GUI Setup ---
def start_game_manual():
    global max_sum, special_seq, deck, agent_hand, bonus
    try:
        max_sum = int(max_sum_entry.get())
        bonus = int(bonus_entry.get())
        seq_values = sorted([int(e.get()) for e in seq_frames])
        special_seq = seq_values
        deck = [(v, s) for s in all_suits for v in range(1, 14)]
        random.shuffle(deck)
        agent_hand = []
        start_frame.pack_forget()
        main_frame.pack()
        update_display()
    except Exception as e:
        print(f"Error starting game: {e}")

def fade_out_animation(step=0, on_finish=None):
    """Recursively fades the label and calls on_finish when done."""
    fade_colors = ["#333333", "#555555", "#777777", "#999999", "#BBBBBB", "#DDDDDD", "#F0F0F0"]
    
    if step < len(fade_colors):
        action_label.config(fg=fade_colors[step])
        root.after(150, lambda: fade_out_animation(step + 1, on_finish))
    else:
        action_label.place_forget()
        # --- Execute the callback function when the animation is complete ---
        if on_finish:
            on_finish()

def show_action_on_gui(action_text, on_finish=None):
    """Displays the action label and starts the fade-out with a completion callback."""
    action_label.config(text=action_text, fg="black")
    action_label.place(relx=0.5, rely=0.7, anchor="center")
    # Start the animation and pass the callback down
    fade_out_animation(on_finish=on_finish)

def run_scripted_step(policy):
    """
    Determines the next action, shows the animation, and then executes the action
    only after the animation is complete.
    """
    global selected_index

    if hand_sum() > max_sum:
        return

    sorted_hand = sorted(agent_hand)
    state_key = tuple(sorted_hand)
    action_str = policy.get(state_key)

    if action_str is None:
        stop_game(f"State {state_key} not in policy file.")
        return

    action = int(action_str)
    def execute_action():
        global selected_index
        if action == 27:
            stop_game("Action 'STOP' from file.")
            return
        elif action == 0:
            pull_card()
        else:
            swap_idx = None
            for i, (v, s) in enumerate(agent_hand):
                card_as_int = v + (13 if s == '♦' else 0)
                if card_as_int == action:
                    swap_idx = i
                    break
            
            if swap_idx is not None:
                selected_index = swap_idx
                swap_card()
            else:
                stop_game(f"Action card '{action}' not in hand.")
                return
        root.after(1500, lambda: run_scripted_step(policy))

    action_text = ""
    if action == 27:
        action_text = "STOP"
    elif action == 0:
        action_text = "PULL"
    else:
        val = (action - 1) % 13 + 1
        suit_char = '♦' if action > 13 else '♥'
        action_text = f"SWAP {val}{suit_char}"
    
    show_action_on_gui(action_text, on_finish=execute_action)

# --- Main Application Setup ---
root = tk.Tk()
root.title("Card Game")

# ... (start_frame setup is unchanged)
start_frame = tk.Frame(root, padx=10, pady=10)
tk.Label(start_frame, text="Max Sum:").grid(row=0, column=0)
max_sum_entry = tk.Entry(start_frame, width=5)
max_sum_entry.grid(row=0, column=1)
tk.Label(start_frame, text="Bonus:").grid(row=1, column=0)
bonus_entry = tk.Entry(start_frame, width=5)
bonus_entry.grid(row=1, column=1)
tk.Label(start_frame, text="Special Sequence:").grid(row=2, column=0)
seq_frame = tk.Frame(start_frame)
seq_frame.grid(row=2, column=1)
seq_frames = [tk.Entry(seq_frame, width=3) for _ in range(3)]
for e in seq_frames: e.pack(side="left")
tk.Button(start_frame, text="Start Game", command=start_game_manual).grid(row=3, columnspan=2, pady=5)

main_frame = tk.Frame(root)
tk.Label(main_frame, text="Deck", font=("Arial", 14)).pack()
deck_canvas = tk.Canvas(main_frame, width=600, height=180, bg="lightgreen")
deck_canvas.pack(pady=5)
tk.Label(main_frame, text="Player's Hand", font=("Arial", 14)).pack()
hand_canvas = tk.Canvas(main_frame, width=600, height=180, bg="lightblue")
hand_canvas.pack(pady=5)
sum_frame = tk.Frame(main_frame)
sum_frame.pack(pady=5)
agent_sum_label = tk.Label(sum_frame, text="Agent's Sum: 0", font=("Arial", 12))
agent_sum_label.pack(side="left", padx=10)
max_sum_label = tk.Label(sum_frame, text="Max Sum: 0", font=("Arial", 12))
max_sum_label.pack(side="left", padx=10)
special_seq_label = tk.Label(sum_frame, text="Special Sequence: []", font=("Arial", 12))
special_seq_label.pack(side="left", padx=10)
button_frame = tk.Frame(main_frame)
button_frame.pack(pady=10)
tk.Button(button_frame, text="Pull", command=pull_card, width=10).grid(row=0, column=0, padx=5)
tk.Button(button_frame, text="Swap", command=swap_card, width=10).grid(row=0, column=1, padx=5)
tk.Button(button_frame, text="Stop", command=lambda: stop_game("Stopped by player"), width=10).grid(row=0, column=2, padx=5)

result_label = tk.Label(main_frame, text="", font=("Arial", 12))
result_label.pack(pady=10)

action_label = tk.Label(main_frame, text="", font=("Arial", 30, "bold"), bg="#F0F0F0")

game_over_label = tk.Label(main_frame, text="", font=("Arial", 36, "bold"), fg="#00008B", bg="#F0F0F0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Card game with manual and scripted modes.")
    parser.add_argument("--policy", type=str, help="Path to input file for scripted play")
    args = parser.parse_args()

    if args.policy:
        try:
            with open(args.policy) as f:
                lines = [line.strip() for line in f if line.strip()]

            max_sum = int(lines[0])
            bonus = int(lines[1])
            special_seq = sorted(list(map(int, lines[2].split())))

            policy = {}
            for line in lines[3:-1]:
                state_str, action_str = map(str.strip, line.split("->"))
                state_card_list = [(int(c[:-1]), '♥' if c[-1].upper() == 'H' else '♦') for c in state_str.split()]
                state_card_list.sort()
                state_tuple = tuple(state_card_list)
                
                policy[state_tuple] = action_str

            initial_hand_str = lines[-1]
            agent_hand = []
            for c in initial_hand_str.split():
                val = int(c[:-1])
                suit = '♥' if (c[-1].upper() == "H") else '♦'
                agent_hand.append((val, suit))

            deck = [c for c in [(v, s) for s in all_suits for v in range(1, 14)] if c not in agent_hand]
            random.shuffle(deck)

            main_frame.pack()
            update_display()

            for child in button_frame.winfo_children():
                child.config(state="disabled")
            root.after(1500, lambda: run_scripted_step(policy))

        except Exception as e:
            print(f"Error processing file: {e}")
            sys.exit(1)
    else:
        start_frame.pack()

    root.mainloop()