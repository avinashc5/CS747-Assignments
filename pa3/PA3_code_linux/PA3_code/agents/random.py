import random
import numpy as np
from .base_agent import BaseAgent
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves

class RandomAgent(BaseAgent):
    # DO NOT CHANGE THIS
    rng = np.random.default_rng(8228)

    def __init__(self, name="RandomAgent"):
        super().__init__(name)

    def move(self, chess_obj):
        moves, proms = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)
        choice_idx = self.rng.choice(range(len(legal_moves)))
        return legal_moves[choice_idx]
