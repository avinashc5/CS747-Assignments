import argparse
import subprocess
import os

def run_command(command, capture_stdout=False):
    """A helper function to run shell commands and handle errors."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        if capture_stdout:
            return result.stdout
    except FileNotFoundError:
        print(f"❌ Error: Command not found. Is '{command.split()[0]}' installed and in your PATH?")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing command: {e.cmd}")
        print(f"   Return Code: {e.returncode}")
        print(f"   Output:\n{e.stderr}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Master script for the MDP card game pipeline.")
    parser.add_argument('--limit', type=int, required=True, help='A positive integer for the maximum sum.')
    parser.add_argument('--bonus', type=int, required=True, help='A positive integer for the bonus score.')
    parser.add_argument('--sequence', type=int, nargs=3, required=True, help='Three consecutive positive integers.')
    parser.add_argument(
        '--initial_state',
        nargs='+',
        required=True,
        help='A space-separated list of initial cards (e.g., 1H 4H 3D).'
    )
    
    args = parser.parse_args()
    
    # --- Sanity checks ---
    if args.limit <= 0:
        parser.error("--limit must be a positive integer.")
    if args.bonus < 0:
        parser.error("--bonus must be a non negative integer.")
    seq = sorted(args.sequence)
    if not (seq[1] == seq[0] + 1 and seq[2] == seq[1] + 1):
        parser.error(f"--sequence values must be consecutive. You provided: {args.sequence}")
    if not all(n > 0 and n < 13 for n in seq):
        parser.error(f"--sequence values must be positive and < 13. You provided: {args.sequence}")
    for card in args.initial_state:
        if len(card) < 2:
            parser.error(f"Invalid card format in --initial_state: '{card}'.")
        val_str, suit = card[:-1], card[-1].upper()
        if suit not in ['H', 'D']:
            parser.error(f"Invalid suit in --initial_state: '{card}'.")
        try:
            val = int(val_str)
            if not (1 <= val <= 13):
                parser.error(f"Invalid value in --initial_state: '{card}'.")
        except ValueError:
            parser.error(f"Invalid value in --initial_state: '{card}'.")
    if len(args.initial_state) != len(set(args.initial_state)):
        parser.error("--initial_state cannot contain duplicate cards.")
    # --- Sanity Check ends ---

    # --- Define filenames ---
    ENCODER_INPUT_FILE = "tmp_encoder_input.txt"
    MDP_FILE = "card_game.txt"
    PLANNER_OUTPUT_FILE = "tmp_planner_output.txt"
    FINAL_POLICY_FILE = "final_policy.txt"
    
    # MODIFIED: TESTCASE_FILE is no longer needed
    temp_files = [ENCODER_INPUT_FILE, MDP_FILE, PLANNER_OUTPUT_FILE]

    try:
        # --- Step 1: Encoding MDP ---
        print("\n--- Step 1: Encoding MDP ---")
        with open(ENCODER_INPUT_FILE, "w") as f:
            f.write("Configuration:\n")
            f.write(f"{args.limit}\n")
            f.write(f"{args.bonus}\n")
            f.write(f"{' '.join(map(str, args.sequence))}\n")
        
        run_command(f"python3 encoder.py --game_config {ENCODER_INPUT_FILE} > {MDP_FILE}")
        print(f"✅ Encoder created '{MDP_FILE}'.")

        # --- Step 2: Planning with value iteration ---
        print("\n--- Step 2: Planning ---")
        planner_output = run_command(f"python3 planner.py --mdp {MDP_FILE}", capture_stdout=True)
        with open(PLANNER_OUTPUT_FILE, "w") as f:
            f.write(planner_output)
        print(f"✅ Planner output saved to '{PLANNER_OUTPUT_FILE}'.")

        # --- MODIFIED: Step 3: Directly decode the full policy ---
        print("\n--- Step 3: Decoding all relevant states ---")
        decoder_output = run_command(
            f"python3 decoder.py --value_policy {PLANNER_OUTPUT_FILE} --automate {ENCODER_INPUT_FILE}",
            capture_stdout=True
        )
        print(f"✅ Decoder generated the complete policy.")
        
        # --- MODIFIED: Step 4: Assemble the final policy file ---
        print("\n--- Step 4: Assembling Final Policy File ---")
        with open(FINAL_POLICY_FILE, "w") as f:
            # Write header
            f.write(f"{args.limit}\n")
            f.write(f"{args.bonus}\n")
            f.write(f"{' '.join(map(str, args.sequence))}\n")
            
            # Write the policy mappings directly from the decoder's output
            f.write(decoder_output)
            
            # Append the initial state to the end of the file, ensuring it's on a new line
            initial_state_line = " ".join(args.initial_state)
            if not decoder_output.endswith('\n'):
                f.write('\n')
            f.write(f"{initial_state_line}\n")

        print(f"Success! Final policy file created at '{FINAL_POLICY_FILE}'.")

        # --- Step 5: Launch the GUI with the generated policy ---
        print(f"\n--- Step 5: Launching GUI ---")
        run_command(f"python3 gui.py --policy {FINAL_POLICY_FILE}")

    finally:
        # --- Cleanup: Remove all temporary files ---
        print("\n--- Cleaning up temporary files ---")
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"🗑️  Removed {file}")

if __name__ == "__main__":
    main()