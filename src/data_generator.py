import numpy as np
import time

# Import all torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def is_win(board):
    # Check all win conditions
    wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8), # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8), # columns
            (0, 4, 8), (2, 4, 6)]            # diagonals
    for a, b, c in wins:
        if board[a] == board[b] == board[c] and board[a] is not None:
            return True
    return False

def is_draw(board):
    return all(square is not None for square in board)

def generate_games(board, depth, games, current_game=[]):
    # Base condition: If game is won or draw
    if is_win(board) or is_draw(board):
        games.append(current_game.copy())
        return

    for i in range(9): # Check each square
        if board[i] is None: # If square is empty
            board[i] = depth % 2 # Place current player's token
            current_game.append(i) # Record move
            
            # Recursive step
            generate_games(board, depth + 1, games, current_game)
            
            # Undo the move for backtracking
            board[i] = None
            current_game.pop()

def tic_tac_toe_games():
    board = [None] * 9 # Initialize empty board
    games = []
    generate_games(board, 0, games)
    return games


if __name__ == "__main__":

    start = time.time()

    # Generate all possible games
    all_games = tic_tac_toe_games()
    print(f"Total games generated: {len(all_games)}")

    # SAVE
    all_games_padded = np.array([game + [9] * (9 - len(game)) for game in all_games])
    print(f"Shape of padded games: {all_games_padded.shape}")
    # Convert to PyTorch tensor (int)
    all_games_tensor = torch.tensor(all_games_padded, dtype=torch.long)
    # Save to data
    torch.save(all_games_tensor, "data/all_games_tensor.pt")
    
    end = time.time()
    print(f"Generated and saved data in {end - start:.2f} seconds")