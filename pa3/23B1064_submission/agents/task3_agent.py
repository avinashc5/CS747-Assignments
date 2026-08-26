from minichess.chess.fastchess import Chess
from .base_agent import BaseAgent
import random
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves
from typing import Tuple

DEPTH = 4

piece_weights = [1, 3, 3, 5, 9, 200]
mobility_weight = 1

def set_bit_count(n: int):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

class Task3Agent(BaseAgent):
    def __init__(self, name="Task3Agent"):
        super().__init__(name)

    def move(self, chess_obj:Chess):
        # get the chess object and return the move to make. the move is a tuple of original position, delta position and promotion to what piece
        
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
            
            return material_score
        
        def minimax(board: Chess, depth: int, current_player: bool, alpha, beta) -> Tuple[int, Tuple[Tuple[int, int], Tuple[int, int], int]]:
            '''
                board: has the whole information of the board
                depth: how deep you want to go
                current_player: true if the player is white, else black
                alpha: alpha value for alpha-beta pruning
                beta: beta value for alpha-beta pruning
                reward tells what white is getting
            '''
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

                evaluation, _ = minimax(board_copy, depth - 1, not current_player, alpha, beta)
                
                if current_player:
                    if evaluation >= best_move[0]:
                        best_move = (evaluation, move)
                    alpha = max(alpha, best_move[0])
                else:
                    if evaluation <= best_move[0]:
                        best_move = (evaluation, move)
                    beta = min(beta, best_move[0])
                
                if beta < alpha:
                    break  # prune
                    
            return best_move
            
        evaluation, move = minimax(chess_obj, DEPTH, chess_obj.turn, -100000, 100000)
        return move


    ### Any other utility functions you want to define for your agent.
    def reset(self,): ...
