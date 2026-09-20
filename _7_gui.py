import pygame
import chess
import torch
import os
from _1_engine import GameState
from _5_mcts import MCTS
from _2_model import AntichessNet

# 1. The Screen Setup 
WIDTH = HEIGHT = 800
SQUARE_SIZE = WIDTH // 8
IMAGES = {}

def load_images():
    """
    Loads images once into a dictionary to save memory.
    Uses your specific 'chess_pieces' folder and 'WP.png', 'BB.png' naming structure.
    """
    piece_map = {
        'P': 'WP', 'R': 'WR', 'N': 'WN', 'B': 'WB', 'Q': 'WQ', 'K': 'WK',
        'p': 'BP', 'r': 'BR', 'n': 'BN', 'b': 'BB', 'q': 'BQ', 'k': 'BK'
    }
    for symbol, img_name in piece_map.items():
        # Look in the "chess_pieces" folder instead of "images"
        image_path = os.path.join("chess_pieces", f"{img_name}.png")
        
        if os.path.exists(image_path):
            # Load the image and scale it to exactly 100x100 pixels to fit the square
            IMAGES[symbol] = pygame.transform.scale(pygame.image.load(image_path), (SQUARE_SIZE, SQUARE_SIZE))
        else:
            print(f"Warning: Could not find {image_path}")

def draw_board(screen, board, selected_square):
    # Draw the alternating light and dark squares 
    colors = [pygame.Color("white"), pygame.Color("gray")]
    
    for row in range(8):
        for col in range(8):
            color = colors[((row + col) % 2)]
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
            # Convert GUI row/col to the chess board square index (0 to 63)
            square = (7 - row) * 8 + col
            
            # Highlight the selected square in light blue
            if selected_square == square:
                pygame.draw.rect(screen, pygame.Color("lightblue"), pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
            # Draw the piece if one exists on this square
            piece = board.piece_at(square)
            if piece and piece.symbol() in IMAGES:
                screen.blit(IMAGES[piece.symbol()], pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def play_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("My Antichess AI")
    load_images()
    
    # 2. Wake up the AI 
    game = GameState()
    model = AntichessNet() 
    
    # Load your trained weights! (We use map_location='cpu' so it works safely on your local machine)
    if os.path.exists("antichess_model.pth"):
        model.load_state_dict(torch.load("antichess_model.pth", map_location=torch.device('cpu')))
        print("Successfully loaded AI Brain!")
    else:
        print("Warning: antichess_model.pth not found. The AI will play randomly.")
        
    model.eval() # Tell PyTorch we are playing, not training
    
    ai_player = MCTS(model)
    running = True
    player_is_white = True # You play as White, AI plays as Black
    selected_square = None # Keeps track of the piece you clicked on
    
    while running:
        draw_board(screen, game.board, selected_square)
        pygame.display.flip()
        
        # Check if the game is over using your engine.py rules
        is_over, outcome = game.check_game_over()
        if is_over:
            print(f"Game Over! Outcome Score: {outcome}")
            running = False
            continue

        # 3. The AI's Turn 
        if game.board.turn != player_is_white:
            print("AI is thinking...")
            # We set temperature=0.0 so the AI exploits its best move instead of exploring
            best_move, _ = ai_player.search(game.board, num_simulations=400, temperature=0.0)
            game.make_move(best_move)
            
        # 4. The Human's Turn 
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                # Listen for the user clicking the mouse 
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Get the exact x, y pixel coordinates of the mouse click 
                    x, y = pygame.mouse.get_pos()
                    
                    # Convert the 800x800 pixels into the 8x8 grid coordinates [1]
                    col = x // SQUARE_SIZE
                    row = 7 - (y // SQUARE_SIZE) # PyGame counts Y from the top down, so we flip it
                    clicked_square = chess.square(col, row)
                    
                    # Click 1: Select a piece
                    if selected_square is None:
                        piece = game.board.piece_at(clicked_square)
                        if piece and piece.color == player_is_white:
                            selected_square = clicked_square
                            
                    # Click 2: Choose destination and make the move
                    else:
                        move = chess.Move(selected_square, clicked_square)
                        
                        # Auto-promote pawns to Queens if they reach the end of the board
                        if game.board.piece_at(selected_square) and game.board.piece_at(selected_square).piece_type == chess.PAWN:
                            if row == 7 or row == 0:
                                move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)
                        
                        # Verify the move follows Antichess forced-capture rules
                        if move in game.board.legal_moves:
                            game.make_move(move)
                            
                        # Deselect the piece whether the move was legal or not
                        selected_square = None

if __name__ == "__main__":
    play_game()