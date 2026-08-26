from minichess.chess.fastchess import Chess
from .base_agent import BaseAgent
import random
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves
from typing import Tuple

DEPTH = 5

piece_weights = [1, 3, 3, 5, 9, 200]
mobility_weight = 1

def set_bit_count(n: int):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

class Task4Agent(BaseAgent):
    def __init__(self, name="Task4Agent"):
        super().__init__(name)

    def move(self, chess_obj:Chess) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
        # uses negamax + alpha beta pruning
        # evaluation based on material + mobility (kinda basic)
        MAX_SCORE = 100000.0

        def material_score(board: Chess):
            s = 0
            for i in range(6):
                w = set_bit_count(int(board.bitboards[1, i]))
                b = set_bit_count(int(board.bitboards[0, i]))
                s += piece_weights[i] * (w - b)
            return s

        def evaluate(board: Chess):
            res = board.game_result()
            if res is not None:
                if res == 0:
                    return 0.0
                elif res == 1:
                    return MAX_SCORE
                else:
                    return -MAX_SCORE
            
            mat = material_score(board)

            # count mobility for both sides
            def count_moves(matrix):
                tot = 0
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        tot += set_bit_count(int(matrix[i, j]))
                return tot

            moves_mat, _ = board.legal_moves()
            my_moves = count_moves(moves_mat)

            orig_turn = board.turn
            board.turn = not orig_turn
            opp_mat, _ = board.legal_moves()
            opp_moves = count_moves(opp_mat)
            board.turn = orig_turn

            mob = my_moves - opp_moves
            return mat + mobility_weight * mob

        def get_legal_moves(board: Chess):
            m, p = board.legal_moves()
            return piece_matrix_to_legal_moves(m, p)

        def order_moves(board: Chess, moves, color):
            scored = []
            for m in moves:
                b = board.copy()
                (i,j),(dx,dy),prom = m
                b.make_move(i,j,dx,dy,prom)
                sc = color * evaluate(b)
                scored.append((sc, m))
            # sort moves (better ones first)
            scored.sort(key=lambda x: x[0], reverse=True)
            res = []
            i = 0
            while i < len(scored):
                j = i + 1
                grp = [scored[i][1]]
                while j < len(scored) and abs(scored[j][0] - scored[i][0]) < 1e-6:
                    grp.append(scored[j][1])
                    j += 1
                if len(grp) > 1: random.shuffle(grp)
                res.extend(grp)
                i = j
            return res

        def quiescence(board: Chess, alpha, beta, color, depth=0):
            # stop if too deep
            if depth > 8:
                return color * evaluate(board)

            stand = color * evaluate(board)
            if stand >= beta:
                return beta
            if alpha < stand:
                alpha = stand

            base = material_score(board)
            moves = get_legal_moves(board)
            captures = []
            for m in moves:
                b = board.copy()
                (i,j),(dx,dy),prom = m
                b.make_move(i,j,dx,dy,prom)
                if abs(material_score(b) - base) > 0.0:
                    captures.append(m)
            if not captures:
                return stand

            captures = order_moves(board, captures, color)
            for m in captures:
                b = board.copy()
                (i,j),(dx,dy),prom = m
                b.make_move(i,j,dx,dy,prom)
                val = -quiescence(b, -beta, -alpha, -color, depth+1)
                if val >= beta:
                    return beta
                if val > alpha:
                    alpha = val
            return alpha

        def negamax(board: Chess, depth, alpha, beta, color):
            res = board.game_result()
            if res is not None:
                if res == 0:
                    return 0.0, None
                elif res == 1:
                    return (MAX_SCORE if color == 1 else -MAX_SCORE), None
                else:
                    return (-MAX_SCORE if color == 1 else MAX_SCORE), None

            if depth == 0:
                val = quiescence(board, alpha, beta, color)
                return val, None

            moves = get_legal_moves(board)
            if not moves:
                return color * evaluate(board), None

            moves = order_moves(board, moves, color)
            best_val = -999999
            best_move = None

            for m in moves:
                b = board.copy()
                (i,j),(dx,dy),prom = m
                b.make_move(i,j,dx,dy,prom)
                sc, _ = negamax(b, depth-1, -beta, -alpha, -color)
                sc = -sc
                if sc > best_val:
                    best_val = sc
                    best_move = m
                if sc > alpha:
                    alpha = sc
                if alpha >= beta:
                    break
            return best_val, best_move

        color = 1 if chess_obj.turn else -1
        val, best = negamax(chess_obj, DEPTH, -MAX_SCORE, MAX_SCORE, color)

        if best is None:
            legal = get_legal_moves(chess_obj)
            if not legal: return None
            return random.choice(legal)
        return best

    ### Any other utility functions you want to define for your agent.
    def reset(self,): ...
