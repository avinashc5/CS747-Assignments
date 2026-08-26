from minichess.chess.fastchess import Chess


class BaseAgent:
    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    def move(self, chess_obj: Chess):
        raise NotImplementedError("Agent must implement move(chess_obj)")

    def reset(self,):
        '''If you maintain internal state of agent then you should implement this function to reset your internal parameters.'''
        ...
        
    def __repr__(self):
        return f"<Agent {self.name}>"
