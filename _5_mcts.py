
import math
import random
import torch
import numpy as np
import chess
from typing import Dict, Tuple, Optional
from _4_encoder import encode_board

class Node:
    def __init__(self, board: chess.Board, parent: Optional['Node'] = None, move_taken: Optional[chess.Move] = None, prior_prob: float = 0.0):
        self.board: chess.Board = board.copy()
        self.parent: Optional['Node'] = parent
        self.move_taken: Optional[chess.Move] = move_taken
        self.children: Dict[chess.Move, 'Node'] = {}

        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior_prob: float = prior_prob

    def is_expanded(self) -> bool:
        return len(self.children) > 0

class MCTS:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def search(self, initial_board: chess.Board, num_simulations: int = 800, temperature: float = 1.0) -> Tuple[chess.Move, np.ndarray]:
        root = Node(initial_board)

        for _ in range(num_simulations):
            node = root

            while node.is_expanded() and not node.board.is_variant_end():
                node = self.select_best_child(node)

            outcome = 0.0
            if not node.board.is_variant_end():
                tensor_board = encode_board(node.board)
                tensor_board = torch.tensor(tensor_board).unsqueeze(0)

                with torch.no_grad():
                    value, policy = self.model(tensor_board)

                value = value.item()
                policy = policy.squeeze(0).numpy()

                legal_moves = list(node.board.legal_moves)
                for i, move in enumerate(legal_moves):
                    node.board.push(move)
                    prob = policy[i] if i < len(policy) else 0.01
                    node.children[move] = Node(node.board, parent=node, move_taken=move, prior_prob=prob)
                    node.board.pop()

                outcome = value
            else:
                if node.board.is_variant_win(): outcome = 1.0
                elif node.board.is_variant_loss(): outcome = -1.0
                else: outcome = 0.0

            while node is not None:
                node.visit_count += 1
                node.value_sum += outcome
                outcome = -outcome
                node = node.parent

        return self.get_most_visited_move(root, temperature), self.get_policy_distribution(root)

    def select_best_child(self, node: Node) -> Node:
        best_score = -float('inf')
        best_child = None
        c_puct = 2.5
        total_visits = sum(child.visit_count for child in node.children.values())

        for move, child in node.children.items():
            if child.visit_count > 0:
                q_value = child.value_sum / child.visit_count
            else:
                q_value = 0.0

            u_value = c_puct * child.prior_prob * (math.sqrt(total_visits) / (1 + child.visit_count))
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def get_most_visited_move(self, root: Node, temperature: float) -> chess.Move:
        moves = list(root.children.keys())
        visits = [child.visit_count for child in root.children.values()]

        # EXPLOITATION (Temp = 0)
        if temperature == 0.0:
            best_move = None
            max_visits = -1
            for move, child in root.children.items():
                if child.visit_count > max_visits:
                    max_visits = child.visit_count
                    best_move = move
            return best_move

        # EXPLORATION (Temp = 1)
        total_visits = sum(visits)
        if total_visits == 0:
            return random.choice(moves)

        probabilities = [v / total_visits for v in visits]

        # THE ACTUAL BUG FIX: Using .pop() to bypass the interface formatting issue!
        selected_moves_list = random.choices(moves, weights=probabilities, k=1)
        return selected_moves_list.pop()

    def get_policy_distribution(self, root: Node) -> np.ndarray:
        total_visits = sum(child.visit_count for child in root.children.values())
        mcts_policy = np.zeros(4216, dtype=np.float32)

        for i, (move, child) in enumerate(root.children.items()):
            if i < 4216:
                mcts_policy[i] = child.visit_count / total_visits if total_visits > 0 else 0.0

        return mcts_policy