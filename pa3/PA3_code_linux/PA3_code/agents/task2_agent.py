from minichess.chess.fastchess import Chess
from .base_agent import BaseAgent
import random
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves
from typing import Tuple

piece_weights = [1, 3, 3, 5, 9, 200]
mobility_weight = 1

def set_bit_count(n: int):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

class Task2Agent(BaseAgent):
    def __init__(self, name="Task2Agent"):
        super().__init__(name)

    def move(self, chess_obj:Chess) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
        # get the chess object and return the move to make. the move is a tuple of original position, delta position and promotion to what piece
        
        def evaluate(board: Chess): # https://www.chessprogramming.org/Evaluation
            res = board.game_result()
            if res != None:
                if res == 0:
                    return 0.0
                elif res == 1:
                    return 100000.0
                else:
                    return -100000.0
            material_score = 0.0
            for i in range(6):
                white_player_count = set_bit_count(int(board.bitboards[1, i]))
                black_player_count = set_bit_count(int(board.bitboards[0, i]))
                material_score += piece_weights[i] * (white_player_count - black_player_count)
            
            curr_moves, curr_proms = board.legal_moves()
            opp_moves, opp_proms = None, None
            
            board_copy = board.copy()
            board_copy.make_null_move()
            opp_moves, opp_proms = board_copy.legal_moves()
            
            curr_mobility_score = len(piece_matrix_to_legal_moves(curr_moves, curr_proms))
            opp_mobility_score = len(piece_matrix_to_legal_moves(opp_moves, opp_proms))
            
            mobility_score = mobility_weight * (curr_mobility_score - opp_mobility_score)
            if board.turn == 1:
                return material_score + mobility_score
            return material_score - mobility_score
        
        def minimax(board: Chess, depth: int, current_player: bool) -> Tuple[int, Tuple[Tuple[int, int], Tuple[int, int], int]]:
            '''
                board: has the whole information of the board
                depth: how deep you want to go
                current_player: true if the player is white, else black
                reward tells what white is getting
            '''
            # Currently implementing it without alpha-beta pruning
            if board.game_result() != None or depth == 0:
                # If you have reached the max depth you had decided to go or the game is over
                return evaluate(board), None
            
            best_move = None
            if current_player: # White. try to get a better move than this. maximize
                best_move = (-100000, None)
            else: # Black. try to get a better move than this. minimize
                best_move = (100000, None)
            
            moves, proms = board.legal_moves()
            legal_moves = piece_matrix_to_legal_moves(moves, proms)
            for move in legal_moves:
                board_copy = board.copy()
                (i, j), (dx, dy), promotion = move
                
                board_copy.make_move(i, j, dx, dy, promotion)
                # If this move ends the game, evaluate the terminal position; otherwise recurse
                if board_copy.game_result() != None:
                    evaluation = evaluate(board_copy)
                else:
                    evaluation, _ = minimax(board_copy, depth - 1, not current_player)
                
                if current_player:
                    if evaluation >= best_move[0]:
                        best_move = (evaluation, move)
                else:
                    if evaluation <= best_move[0]:
                        best_move = (evaluation, move)
            return best_move
            
        evaluation, move = minimax(chess_obj, 3, chess_obj.turn)
        return move

    ### Any other utility functions you want to define for your agent.
    def reset(self,): ...
