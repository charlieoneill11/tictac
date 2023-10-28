import pickle

def check_win(board, player):
    winning_moves = []
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
                            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
                            (0, 4, 8), (2, 4, 6)]  # diagonals
    for a, b, c in winning_combinations:
        if board[a] == board[b] == player and board[c] == ' ':
            winning_moves.append(c)
        if board[a] == board[c] == player and board[b] == ' ':
            winning_moves.append(b)
        if board[b] == board[c] == player and board[a] == ' ':
            winning_moves.append(a)
    return winning_moves

def check_block(board, player):
    opponent = 'O' if player == 'X' else 'X'
    return check_win(board, opponent)

def is_unblocked_line(board, player, a, b, c):
    """
    Check if a line (defined by indices a, b, c) is unblocked for the given player.
    """
    return ((board[a] == board[b] == player and board[c] == ' ') or
            (board[a] == board[c] == player and board[b] == ' ') or
            (board[b] == board[c] == player and board[a] == ' '))

def check_fork(board, player):
    fork_moves = []
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
                            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
                            (0, 4, 8), (2, 4, 6)]  # diagonals
    for i in range(9):
        if board[i] == ' ':
            board[i] = player  # Temporarily place the player's marker
            unblocked_lines = 0  # Count of unblocked lines
            for a, b, c in winning_combinations:
                if is_unblocked_line(board, player, a, b, c):
                    unblocked_lines += 1
            if unblocked_lines >= 2:
                fork_moves.append(i)
            board[i] = ' '  # Reset the board
    return fork_moves

def is_two_in_a_row(board, player, a, b, c):
    return ((board[a] == player and board[b] == player and board[c] == ' ') or
            (board[a] == player and board[c] == player and board[b] == ' ') or
            (board[b] == player and board[c] == player and board[a] == ' '))

def check_block_fork(board, player):
    opponent = 'O' if player == 'X' else 'X'
    forks = check_fork(board, opponent)

    # Check if opponent has two opposite corners
    if board[0] == board[8] == opponent:
        if board[2] == ' ' or board[6] == ' ':
            to_rtn = []
            for i in [1, 3, 5, 7]:
                if board[i] == ' ': to_rtn.append(i)
            return to_rtn
    if board[2] == board[6] == opponent:
        if board[0] == ' ' or board[8] == ' ':
            to_rtn = []
            for i in [1, 3, 5, 7]:
                if board[i] == ' ': to_rtn.append(i)
            return to_rtn

    if len(forks) > 1:
        two_in_a_row_moves = []
        winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
                                (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
                                (0, 4, 8), (2, 4, 6)]  # diagonals
        for i in range(9):
            if board[i] == ' ':
                board[i] = player  # temporarily place the player's marker
                for a, b, c in winning_combinations:
                    if is_two_in_a_row(board, player, a, b, c):
                        two_in_a_row_moves.append(i)
                        break  # no need to check further combinations for this move
                board[i] = ' '  # reset the board
        # return intersection of two_in_a_row_moves and forks
        return list(set(two_in_a_row_moves) & set(forks))
    if len(forks) == 1:
        return forks
    return []

def check_center(board):
    return [4] if board[4] == ' ' else []

def check_opposite_corner(board, player):
    opponent = 'O' if player == 'X' else 'X'
    opposite_corners = [(0, 8), (2, 6), (6, 2), (8, 0)]
    moves = []
    for a, b in opposite_corners:
        if board[a] == opponent and board[b] == ' ':
            moves.append(b)
        if board[b] == opponent and board[a] == ' ':
            moves.append(a)
    return moves

def check_empty_corner(board):
    corners = [0, 2, 6, 8]
    return [corner for corner in corners if board[corner] == ' ']

def check_empty_side(board):
    sides = [1, 3, 5, 7]
    return [side for side in sides if board[side] == ' ']

def get_optimal_moves(board, player):
    for check in [check_win, check_block, check_fork, check_block_fork]:
        moves = check(board, player)
        if moves:
            return moves

    for check in [check_center, check_empty_corner, check_empty_side]:
        moves = check(board)
        if moves:
            return moves

    for check in [check_opposite_corner]:
        moves = check(board, player)
        if moves:
            return moves

    return []  # should never reach this point in a valid game

finished_games = []

def is_winner(board, player):
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
                            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
                            (0, 4, 8), (2, 4, 6)]  # diagonals
    for a, b, c in winning_combinations:
        if board[a] == board[b] == board[c] == player:
            return True
    return False

def simulate_game(board, move_sequence, next_player):
    # check for game over conditions (win or draw)
    if is_winner(board, 'X') or is_winner(board, 'O'):
        finished_games.append(move_sequence[:])
        return
    if ' ' not in board:
        finished_games.append(move_sequence[:])
        return
    
    # optimal player's move
    if next_player == 'X':
        optimal_moves = get_optimal_moves(board, next_player)
        for move in optimal_moves:
            board[move] = next_player  # Make the move
            move_sequence.append(move)  # Record the move
            simulate_game(board, move_sequence, 'O')  # Recursive call
            board[move] = ' '  # Undo the move
            move_sequence.pop()  # Remove the last move from the sequence
    
    # all moves for the non-optimal player
    else:
        for move in range(9):
            if board[move] == ' ':
                board[move] = next_player  # Make the move
                move_sequence.append(move)  # Record the move
                simulate_game(board, move_sequence, 'X')  # Recursive call
                board[move] = ' '  # Undo the move
                move_sequence.pop()  # Remove the last move from the sequence

initial_board = [' ' for _ in range(9)]
initial_move_sequence = []
print('Simulating games where X goes first...')
simulate_game(initial_board, initial_move_sequence, 'X')



# Initialize list to store sequences of all finished games
finished_games_O_first = []
o_wins = 0

def simulate_game_O_first(board, move_sequence):
    """
    Simulate a game of Tic-Tac-Toe recursively where 'O' goes first.
    
    Parameters:
        board (list): The current game board.
        move_sequence (list): The sequence of moves made so far.
    """
    if is_winner(board, 'X') or is_winner(board, 'O'):
        finished_games_O_first.append(move_sequence[:])
        return
    if ' ' not in board:
        finished_games_O_first.append(move_sequence[:])
        return
    
    # all moves for the non-optimal player
    for move in range(9):
        if board[move] == ' ':
            board[move] = 'O' 
            move_sequence.append(move) 

            # optimal player's move
            optimal_moves = get_optimal_moves(board, 'X')
            for x_move in optimal_moves:
                board[x_move] = 'X'  
                move_sequence.append(x_move) 
                simulate_game_O_first(board, move_sequence)
                board[x_move] = ' '  
                move_sequence.pop() 
                
            board[move] = ' ' 
            move_sequence.pop()

# initialise board and move_sequence
initial_board = [' ' for _ in range(9)]
initial_move_sequence = []

# start the simulation with 'O' going first
print('Simulating games where O goes first...')
simulate_game_O_first(initial_board, initial_move_sequence)


# Save to pickle
print('Saving to pickle...')
with open('data/finished_games_X_first.pickle', 'wb') as f:
    pickle.dump(finished_games, f)

with open('data/finished_games_O_first.pickle', 'wb') as f:
    pickle.dump(finished_games_O_first, f)

print('Done.')