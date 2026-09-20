# 4. The Translator (encoder.py)

import numpy as np
import chess

def encode_board(board: chess.Board) -> np.ndarray:
    encoded = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type - 1
            if piece.color == chess.BLACK:
                piece_type += 6

            row = chess.square_rank(square)
            col = chess.square_file(square)
            encoded[piece_type][row][col] = 1.0

    if board.turn == chess.BLACK:
        encoded = np.flip(encoded, axis=(1, 2)).copy()

    return encoded
