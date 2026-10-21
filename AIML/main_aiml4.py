import math

def check_winner(b, player):
    win_states = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    return any(all(b[pos] == player for pos in state) for state in win_states)

def minimax(b, is_max):
    if check_winner(b, 'O'):
        return 1
    if check_winner(b, 'X'):
        return -1
    if ' ' not in b:
        return 0

    if is_max:
        best = -math.inf
        for i in range(9):
            if b[i] == ' ':
                b[i] = 'O'
                best = max(best, minimax(b, False))
                b[i] = ' '
        return best
    else:
        best = math.inf
        for i in range(9):
            if b[i] == ' ':
                b[i] = 'X'
                best = min(best, minimax(b, True))
                b[i] = ' '
        return best

def find_best_move(b):
    best_val = -math.inf
    move = -1
    for i in range(9):
        if b[i] == ' ':
            b[i] = 'O'
            move_val = minimax(b, False)
            b[i] = ' '
            if move_val > best_val:
                best_val = move_val
                move = i
    return move

def print_board(b):
    for i in range(3):
        print(f"| {b[i * 3]} | {b[i * 3 + 1]} | {b[i * 3 + 2]} |")
    print()

board = [' ' for _ in range(9)]
print("Tic-Tac-Toe: Human (X) vs Unbeatable AI (O)")
print("Positions: 0-8 starting top-left to bottom-right\n")

while ' ' in board:
    print_board(board)
    try:
        move = int(input("Enter your move (0-8): "))
        if move < 0 or move > 8:
            print("Invalid index! Enter a number between 0 and 8.")
            continue
        if board[move] != ' ':
            print("Position taken!")
            continue
    except (ValueError, IndexError):
        print("Invalid input!")
        continue

    board[move] = 'X'
    if check_winner(board, 'X'):
        print_board(board)
        print("Human wins!")
        break

    if ' ' not in board:
        break

    ai_move = find_best_move(board)
    if ai_move != -1:
        board[ai_move] = 'O'

    if check_winner(board, 'O'):
        print_board(board)
        print("AI wins!")
        break

if not check_winner(board, 'X') and not check_winner(board, 'O'):
    print_board(board)
    print("It's a draw!")
