import time
import random
import numpy as np
from typing import Tuple
from minichess.chess.chess_helpers import get_initial_chess_object
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

from agents.random import RandomAgent
from agents.task1_agent import Task1Agent
from agents.task2_agent import Task2Agent
from agents.task3_agent import Task3Agent

from agents.rational_agent import RationalAgent

import warnings
# warnings.filterwarnings('ignore')

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NUM_GAMES = 100
BOARD_TYPE = '5x4microchess' 

TIME_THRESHOLDS = [0.005, 0.1, 0.2]
POINT_THRESHOLDS = [27, 20, 50]

def play_matches(agent1, agent2) -> Tuple[dict, dict]:
    stats = {
        agent1.name: {"wins_white": 0, "wins_black": 0, "total_wins": 0, "total_time": 0.0, "moves": 0, "avg_time":0.0},
        agent2.name: {"wins_white": 0, "wins_black": 0, "total_wins": 0, "total_time": 0.0, "moves": 0, "avg_time":0.0},
        "draws": 0,
        "avg_game_length": 0.0,
    }
    global NUM_GAMES, BOARD_TYPE

    total_moves_all_games = 0
    
    all_game_fens = []
    for g in tqdm(range(1, NUM_GAMES + 1), desc=f"Playing {agent1.name} vs {agent2.name}.", total=NUM_GAMES):
        
        game_fens = []
        if g % 2 == 0 :
            white_agent, black_agent = agent1, agent2
        else:
            white_agent, black_agent = agent2, agent1
        if hasattr(white_agent, 'reset') and callable(white_agent.reset):
            white_agent.reset()
        if hasattr(black_agent, 'reset') and callable(black_agent.reset):
            black_agent.reset()
        
        chess = get_initial_chess_object(BOARD_TYPE)
        move_count = 0
       
        time_white, time_black = 0.0, 0.0
        num_white_moves, num_black_moves = 0, 0
        game_fens.append(chess.fen())
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'overflow encountered in ulong_scalars')
            noviol = True
            while chess.game_result() is None:
                current_agent = white_agent if chess.turn == 1 else black_agent

                start = time.perf_counter()
                mv = current_agent.move(chess.copy())
                elapsed = time.perf_counter() - start

                moves, proms = chess.legal_moves()
                legal_moves = piece_matrix_to_legal_moves(moves, proms)
                if mv is None or mv not in legal_moves:
                    print("     \033[2;31;41m ILLEGAL MOVE ATTEMPTED. \033[0;0m Player looses the game. ")
                    noviol = False
                    violres = -1 if chess.turn == 1 else 1
                    break

                if chess.turn == 1:
                    time_white += elapsed
                    num_white_moves += 1
                else:
                    time_black += elapsed
                    num_black_moves += 1

                (i, j), (dx, dy), promo = mv
                chess.make_move(i, j, dx, dy, promo)

                move_count += 1
                game_fens.append(chess.fen())

        game_fens.append(str(chess.game_result() if noviol else violres))
        all_game_fens.append(game_fens)
        
        result = chess.game_result() if noviol else violres
        total_moves_all_games += move_count
        if result == 1:  # white wins
            stats[white_agent.name]["total_wins"] += 1
            stats[white_agent.name]["wins_white"] += 1
            res = 'white'
        elif result == -1:  # black wins
            stats[black_agent.name]["total_wins"] += 1
            stats[black_agent.name]["wins_black"] += 1
            res = 'black'
        else:
            stats["draws"] += 1
            res = 'draw'

        if num_white_moves > 0:
            stats[white_agent.name]["total_time"] += time_white
            stats[white_agent.name]["moves"] += num_white_moves
        if num_black_moves > 0:
            stats[black_agent.name]["total_time"] += time_black
            stats[black_agent.name]["moves"] += num_black_moves

        if save_fens:
            if res=='black' or res=='white' or res=='draw':
                if not os.path.exists(f"fens/{white_agent.name}_vs_{black_agent.name}/"):
                    os.makedirs(f"fens/{white_agent.name}_vs_{black_agent.name}/")
                game_filename = f"fens/{white_agent.name}_vs_{black_agent.name}/{g}_result_{res}.fen"
                with open(game_filename, "w") as f:
                    for fen in game_fens:
                        f.write(fen + "\n")

    avg_game_len = total_moves_all_games /NUM_GAMES if NUM_GAMES > 0 else 0
    stats["avg_game_length"] = avg_game_len//2
    stats["avg_game_length_plies"] = avg_game_len

    print("=== FINAL RESULTS ===")
    for agent in [agent1.name, agent2.name]:
        s = stats[agent]
        avg_time = s["total_time"] / s["moves"] if s["moves"] > 0 else 0
        s['avg_time'] = avg_time
        print(f"{agent}:")
        print(f"  Wins as White: {s['wins_white']}")
        print(f"  Wins as Black: {s['wins_black']}")
        print(f"  Total Wins:    {s['total_wins']}")
        print(f"  Avg Time for each move: {avg_time:.6f} sec")
    print(f"Draws: {stats['draws']}")
    print(f"Avg Game Length (full moves): {avg_game_len//2:.1f} full moves")
    print(f"Avg Game Length (half moves): {avg_game_len:.1f} plies")

    with open(f"results/{agent1.name}_vs_{agent2.name}.json","w") as f:
        json.dump(stats, f, indent=4)

    return stats

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", default=0, type=int, help="Give the task number which you want to test {1,2,3}. If all tasks then 0.")
    parser.add_argument("--save_fens", action='store_true', help="To save all the game plays as fen files. For later visualization")
    parser.add_argument("--num_games", default=100, help="Number of games to play. Default 100")
    
    args = parser.parse_args()
    task_no = args.task
    save_fens= args.save_fens
    NUM_GAMES = int(args.num_games)

    rand = RandomAgent()
    rational1 = RationalAgent()
    rational2 = RationalAgent()

    if task_no == 0:
        test_agent_1 = Task1Agent()
        result_1 = play_matches(test_agent_1, rand)
        task_1_score = result_1[test_agent_1.name]["total_wins"] - result_1[rand.name]["total_wins"]
        print(f"TASK-1 score: {task_1_score}")
        if task_1_score < POINT_THRESHOLDS[0]:
            print(f"        \033[2;31;41m FAILED \033[0;0m : {task_1_score} < {POINT_THRESHOLDS[0]}")
        else:
            print(f"        \033[2;32;47m PASSED \033[0;0m")
        print()
        if result_1[test_agent_1.name]["avg_time"] > TIME_THRESHOLDS[0]:
                print(f"\033[38;5;208mWARNING: Your average time per turn exceeded threshold <{TIME_THRESHOLDS[0]}s \033[0;0m")
        print()
            
        test_agent_2 = Task2Agent()
        result_2 = play_matches(test_agent_2, rational1)
        task_2_score = result_2[test_agent_2.name]["total_wins"] - result_2[rational1.name]["total_wins"]
        print(f"TASK-2 score: {task_2_score}")
        if task_2_score < POINT_THRESHOLDS[1]:
            print(f"        \033[2;31;41m FAILED \033[0;0m : {task_2_score} < {POINT_THRESHOLDS[1]}")
        else:
            print(f"        \033[2;32;47m PASSED \033[0;0m")
        print()
        if result_2[test_agent_2.name]["avg_time"] > TIME_THRESHOLDS[1]:
                print(f"\033[38;5;208mWARNING: Your average time per turn exceeded threshold <{TIME_THRESHOLDS[1]}s \033[0;0m")
        print()
        
        rational1.reset()

        test_agent_3 = Task3Agent()
        result_3 = play_matches(test_agent_3, rational2)
        task_3_score = result_3[test_agent_3.name]["total_wins"] - result_3[rational2.name]["total_wins"]
        print(f"TASK-1 score: {task_3_score}")
        if task_3_score < POINT_THRESHOLDS[2]:
            print(f"        \033[2;31;41m FAILED \033[0;0m : {task_3_score} < {POINT_THRESHOLDS[2]}")
        else:
            print(f"        \033[2;32;47m PASSED \033[0;0m")
        print()
        if result_3[test_agent_3.name]["avg_time"] > TIME_THRESHOLDS[2]:
                print(f"\033[38;5;208mWARNING: Your average time per turn exceeded threshold <{TIME_THRESHOLDS[2]}s \033[0;0m")
        print()
        
        rational2.reset()

    else:
        if task_no == 1:
            test_agent = Task1Agent()
            result = play_matches(test_agent, rand)
        elif task_no == 2:
            test_agent = Task2Agent()
            result = play_matches(test_agent, rational1)
        elif task_no == 3:
            test_agent = Task3Agent()
            result = play_matches(test_agent, rational2)
            
        if result[test_agent.name]["avg_time"] > TIME_THRESHOLDS[task_no-1]:
            print(f"\033[38;5;208mWARNING: Your average time per turn exceeded threshold <{TIME_THRESHOLDS[task_no-1]} \033[0;0m")
        
        task_score = result[test_agent.name]["total_wins"] - result[(rand if task_no==1 else rational1).name]["total_wins"]
        print(f"TASK-{task_no} score: {task_score}")
        if task_score < POINT_THRESHOLDS[task_no-1]:
            print(f"        \033[2;31;41m FAILED \033[0;0m : {task_score} < {POINT_THRESHOLDS[task_no-1]}")
        else:
            print(f"        \033[2;32;47m PASSED \033[0;0m")
        print()
        

    
