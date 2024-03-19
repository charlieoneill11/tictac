import torch
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_player_positions(board_state, player):
    return [(pos % 3, pos // 3) for pos in board_state if pos % 2 == player]

def add_player_annotations(fig, positions, player_symbol):
    for x, y in positions:
        fig.add_annotation(x=x, y=y, text=player_symbol, showarrow=False,
                           font=dict(size=40), xref="x", yref="y")
        
def add_token_circles(fig, positions):
    for x, y in positions:
        fig.add_shape(type="circle",
                      xref="x", yref="y",
                      x0=x - 0.25, y0=y - 0.25, x1=x + 0.25, y1=y + 0.25,
                      line_color="green", fillcolor="green")

def plot_board_with_logits(board_state, step_logits, layer):
    # Set everything up
    target = board_state[0, layer+1]
    board_state = board_state[0, :layer+1]
    step_logits = step_logits[layer, :]

    # Take softmax BEFORE we discard the game over token
    step_logits = F.softmax(step_logits, dim=-1)
    # Game over probability is the last value in the logits tensor
    game_over_prob = step_logits[-1]
    # Get the predicted next token
    next_token = step_logits.argmax().item()
    # Assume 'layer' is the current step in the game represented by board_state
    step_logits = step_logits[:-1]  # Exclude the last value if not part of the board
    # Reshape into 3x3
    step_logits = step_logits.reshape(3, 3)
    # Imshow with plotly
    fig = px.imshow(step_logits.cpu().detach().numpy(), text_auto=True)

    # Get 'X' and 'O' positions
    x_positions = get_player_positions(board_state[:layer+1], player=0)
    o_positions = get_player_positions(board_state[:layer+1], player=1)

    # Add circles for 'X' and 'O' positions on green tokens
    all_positions = x_positions + o_positions
    add_token_circles(fig, all_positions)

    # Add 'X' and 'O' annotations to the figure
    add_player_annotations(fig, x_positions, 'X')
    add_player_annotations(fig, o_positions, 'O')

    fig.update_xaxes(side="top")  # This will put the (0,0) position of imshow in the top left corner
    fig.update_traces(texttemplate="%{text}", textfont_size=20)  # Set text size

    # Update axes properties to not show any labels or ticks
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    fig.update_layout(autosize=False, width=500, height=500, title=f"Predicted = {next_token} (targ = {target}), Game over prob = {game_over_prob.item()*100:.2f}%")
    fig.show()

def plot_tic_tac_toe_on_tokens(game_sequence):
    # Initialize the board as a list of None
    board = [None] * 9
    
    # Process the game sequence
    for idx, move in enumerate(game_sequence):
        player = 'X' if idx % 2 == 0 else 'O'
        board[move] = player
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add Tic-Tac-Toe grid lines
    for i in range(1, 3):
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=3, line=dict(color="black", width=2))
        fig.add_shape(type="line", y0=i, x0=0, y1=i, x1=3, line=dict(color="black", width=2))
    
    # Add tokens with muted colors as circles
    token_colors = {'X': 'rgba(119, 158, 203, 0.7)', 'O': 'rgba(244, 204, 204, 0.7)'}
    
    # Add marks ('X' or 'O') as annotations on top of tokens
    for i, player in enumerate(board):
        if player:
            # Draw token circle
            fig.add_shape(type="circle",
                          xref="x", yref="y",
                          x0=(i % 3) + 0.1, y0=2 - (i // 3) + 0.1,
                          x1=(i % 3) + 0.9, y1=2 - (i // 3) + 0.9,
                          fillcolor=token_colors[player],
                          line_color=token_colors[player])
            
            # Draw player text
            fig.add_annotation(x=(i % 3) + 0.5, y=2 - (i // 3) + 0.5,
                               text=player, showarrow=False,
                               xanchor="center", yanchor="middle",
                               font=dict(size=40, color="black"))
    
    # Update axes properties
    fig.update_xaxes(range=[0, 3], showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[0, 3], showticklabels=False, showgrid=False, zeroline=False)

    # Update figure properties
    fig.update_layout(height=600, width=600, margin=dict(l=0, r=0, b=0, t=40), plot_bgcolor='rgba(0,0,0,0)')
    
    # Set the aspect ratio to be square
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    # Show the figure
    fig.show()

def plot_tic_tac_toe_boards(game_sequences):
    # Create a 3x3 subplot grid
    fig = make_subplots(rows=3, cols=3, subplot_titles=[f"Game {i+1}" for i in range(9)])
    
    # Token colors
    token_colors = {'X': 'rgba(119, 158, 203, 0.7)', 'O': 'rgba(244, 204, 204, 0.7)'}
    
    # Iterate over each game sequence and each subplot
    for idx, game_sequence in enumerate(game_sequences):
        # Initialize the board as a list of None
        board = [None] * 9
        
        # Process the game sequence
        for move_idx, move in enumerate(game_sequence):
            player = 'X' if move_idx % 2 == 0 else 'O'
            board[move] = player
        
        # Determine the current subplot row and column
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        
        # Add Tic-Tac-Toe grid lines to the subplot
        for i in range(1, 3):
            fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=3, line=dict(color="black", width=2), row=row, col=col)
            fig.add_shape(type="line", y0=i, x0=0, y1=i, x1=3, line=dict(color="black", width=2), row=row, col=col)
        
        # Add tokens and marks to the subplot
        for i, player in enumerate(board):
            if player:
                fig.add_shape(type="circle", xref="x", yref="y",
                              x0=(i % 3) + 0.1, y0=2 - (i // 3) + 0.1,
                              x1=(i % 3) + 0.9, y1=2 - (i // 3) + 0.9,
                              fillcolor=token_colors[player], line_color=token_colors[player],
                              row=row, col=col)
                
                fig.add_annotation(x=(i % 3) + 0.5, y=2 - (i // 3) + 0.5,
                                   text=player, showarrow=False,
                                   xanchor="center", yanchor="middle",
                                   font=dict(size=40, color="black"),
                                   row=row, col=col)
    
    # Update axes properties for all subplots
    fig.update_xaxes(range=[0, 3], showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[0, 3], showticklabels=False, showgrid=False, zeroline=False)
    
    # Update layout to make sure our subplots maintain their aspect ratio
    fig.update_layout(height=900, width=900, showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)  # Apply to one subplot for consistent scaling
    
    # Show the figure
    fig.show()