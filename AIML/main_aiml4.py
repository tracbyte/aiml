import math

# Initialize the board as a flat list of 9 empty spots
board = [' ' for _ in range(9)]

def print_board():
    """Renders the game board to the console."""
    for i in range(3):
        row = board[i*3:(i+1)*3]
        print('| ' + ' | '.join(row) + ' |')
    print()

def check_winner(b, player):
    """Checks if the specified player has won the game."""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    return any(all(b[pos] == player for pos in condition) for condition in win_conditions)

def is_board_full(b):
    """Checks if there are no empty spots left on the board."""
    return ' ' not in b

def minimax(b, depth, is_maximizing):
    """
    Core Minimax algorithm evaluating all potential futures.
    AI is the Maximizer ('O'), Human is the Minimizer ('X').
    """
    # Terminal states: Evaluate utilities
    if check_winner(b, 'O'):
        return 1
    if check_winner(b, 'X'):
        return -1
    if is_board_full(b):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if b[i] == ' ':
                b[i] = 'O'  # Simulate AI move
                score = minimax(b, depth + 1, False)
                b[i] = ' '  # Undo move
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if b[i] == ' ':
                b[i] = 'X'  # Simulate Human move
                score = minimax(b, depth + 1, True)
                b[i] = ' '  # Undo move
                best_score = min(score, best_score)
        return best_score

def find_best_move():
    """Iterates through available moves to pick the highest scoring option."""
    best_score = -math.inf
    move = -1
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(board, 0, False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                move = i
    return move

def play_game():
    """Main game loop managing turns."""
    print("Welcome to Tic-Tac-Toe against Minimax AI!")
    print("Positions are indexed from 0 to 8 (Top-left to Bottom-right):")
    print("| 0 | 1 | 2 |\n| 3 | 4 | 5 |\n| 6 | 7 | 8 |\n")
    
    while True:
        # Human Turn
        print_board()
        try:
            human_move = int(input("Enter your move (0-8): "))
            if board[human_move] != ' ':
                print("Position already taken! Try again.")
                continue
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number between 0 and 8.")
            continue
            
        board[human_move] = 'X'
        
        if check_winner(board, 'X'):
            print_board()
            print("Incredible! You beat the AI!")
            break
        if is_board_full(board):
            print_board()
            print("It's a tie!")
            break
            
        # AI Turn
        print("AI is calculating the optimal move...")
        ai_move = find_best_move()
        if ai_move != -1:
            board[ai_move] = 'O'
            
        if check_winner(board, 'O'):
            print_board()
            print("AI wins! The Minimax algorithm played perfectly.")
            break
        if is_board_full(board):
            print_board()
            print("It's a tie!")
            break

if __name__ == "__main__":
    play_game()

