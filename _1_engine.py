
import chess
import chess.variant

class GameState:
    """Acts as the referee, managing the board and the rules of Antichess."""

    def __init__(self):
        # Sets up the standard 8x8 grid with Antichess forced-capture rules
        self.board = chess.variant.AntichessBoard()

    def make_move(self, move):
        """Pushes a move to the board and automatically swaps the turn."""
        self.board.push(move)

    def get_legal_moves(self) -> list:
        """Generates all legal moves, automatically enforcing forced captures."""
        return list(self.board.legal_moves)

    def check_game_over(self) -> tuple:
        """Checks if the game is over. Returns (Is_Game_Over, Winner_Score)."""
        if self.board.is_game_over() or self.board.is_variant_end():

            if self.board.is_variant_win():
                return True, 1   # Win
            elif self.board.is_variant_loss():
                return True, -1  # Loss
            else:
                return True, 0   # Draw

        return False, None