board = [" " for i in range(9)]

def print_board():
    print(board[0], "|", board[1], "|", board[2])
    print("--+---+--")
    print(board[3], "|", board[4], "|", board[5])
    print("--+---+--")
    print(board[6], "|", board[7], "|", board[8])

def check_winner(symbol):
    win_pos = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for pos in win_pos:
        if board[pos[0]] == board[pos[1]] == board[pos[2]] == symbol:
            return True
    return False

turn = "X"
move_count = 0

while True:
    print_board()
    try:
        choice = int(input(f"Player {turn}, enter position (1-9): ")) - 1
    except:
        print("Invalid input! Please enter number between 1-9.")
        continue

    if choice < 0 or choice > 8 or board[choice] != " ":
        print("Invalid move, try again!")
        continue

    board[choice] = turn
    move_count += 1

    if check_winner(turn):
        print_board()
        print("Player", turn, "won the match!")
        break

    if move_count == 9:
        print_board()
        print("Match Draw!")
        break

    if turn == "X":
        turn = "O"
    else:
        turn = "X"
