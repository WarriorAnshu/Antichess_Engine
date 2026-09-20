# 6. The Training Camp Executable (self_play.py)
# (Note: I added an execution block at the very bottom of this script for you, so that when you type !python self_play.py, it will actually trigger the training loop and save your model!)
import chess
import torch
import numpy as np
from typing import List, Tuple
from _1_engine import GameState
from _4_encoder import encode_board
from _5_mcts import MCTS

class SelfPlayCamp:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def play_training_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        game = GameState()
        mcts = MCTS(self.model)
        game_history = []

        while True:
            best_move, mcts_policy = mcts.search(game.board, num_simulations=64)
            board_tensor = encode_board(game.board)
            current_player = game.board.turn

            game_history.append([board_tensor, mcts_policy, current_player])
            game.make_move(best_move)

            is_over, outcome_score = game.check_game_over()
            if is_over:
                return self.assign_rewards(game_history, outcome_score, game.board.turn)

    def assign_rewards(self, game_history: list, outcome_score: float, final_turn: bool) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        training_data = []

        if outcome_score == 0:
            winner_is_white = None
        else:
            winner_is_white = (final_turn == chess.WHITE and outcome_score == 1) or \
                              (final_turn == chess.BLACK and outcome_score == -1)

        for board_tensor, mcts_policy, player_turn in game_history:
            if outcome_score == 0:
                step_reward = 0.0
            else:
                if player_turn == chess.WHITE:
                    step_reward = 1.0 if winner_is_white else -1.0
                else:
                    step_reward = -1.0 if winner_is_white else 1.0

            training_data.append((board_tensor, mcts_policy, step_reward))

        return training_data

    def generate_batch(self, num_games: int = 100) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        print(f"Starting Training Camp: Playing {num_games} games...")
        batch_data = []

        for i in range(num_games):
            game_data = self.play_training_game()
            batch_data.extend(game_data)
            print(f"Game {i+1}/{num_games} finished! Generated {len(game_data)} positions.")

        return batch_data

if __name__ == "__main__":
    from model import AntichessNet
    from train import train_loop
    import os

    print("Initializing the AI Brain...")
    brain = AntichessNet()

    # Load existing brain if it exists to continue from where we left off
    if os.path.exists('antichess_model.pth'):
        brain.load_state_dict(torch.load('antichess_model.pth'))
        print("Loaded existing brain!")

    camp = SelfPlayCamp(brain)

    # --- THE OUTER LOOP (GENERATIONS) ---
    num_generations = 10
    games_per_gen = 10

    for generation in range(num_generations):
        print(f"\n========== STARTING GENERATION {generation + 1}/{num_generations} ==========")

        # 1. Generate games
        training_data = camp.generate_batch(num_games=games_per_gen)

        # 2. Train the Brain
        print("Passing data to the Teacher...")
        train_loop(training_data, epochs=15, batch_size=256)

        # 3. Save the improved Brain
        torch.save(brain.state_dict(), 'antichess_model.pth')
        print(f"Brain improved and saved for Generation {generation + 1}!")

    print("\nAll generations complete! Training camp finished.")
