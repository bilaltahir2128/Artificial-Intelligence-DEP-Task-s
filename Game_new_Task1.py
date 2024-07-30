def won(board,player):
    for row in range(3):
        if all([board[row][col]==player for col in range(3)]):
            return True
    for col in range(3):
        if all([board[row][col]==player for row in range (3)]):
            return True
    if all([board [i][i]==player for i in range (3)])or all([board[i][2-i]==player for i in range (3)]):
            return True
    return False

def full(board):
    return all([box !=' ' for row in board for box in row])

def minimax(board,depth,is_max):
    if won(board,'O'):
        return 1 #computer
    if won(board,'X'):
        return -1
    if full(board):
        return 0
    if is_max:
        best_score=-float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j]==' ':
                    board[i][j]='O'
                    score =minimax(board,depth,False)
                    board[i][j]=' '
                    best_score=max(score,best_score)
        return best_score    
    else:
        best_score=float('inf')
        for i in range (3):
            for j in range (3):
                if board[i][j]==' ':
                    board[i][j]='X'
                    score=minimax(board,depth,True)
                    board[i][j]=' '
                    best_score=min(score,best_score)
        return best_score

def best_move(board):
    best_score=-float('inf')
    move=(-1,-1)
    for i in range (3):
        for j in range (3):
           if  board[i][j]==' ':
            board[i][j] ='O'
            score=minimax(board,0,False)
            board[i][j] =' '
            if score>best_score:
                best_score=score
                move=(i,j)
    return move

def print_board(board):
    print("  1   2   3")
    for i in range(3):
        print(f"{i+1} {' | '.join(board[i])}")
        if i<2:
            print(" ---|---|---")

def play():
    board=[[' ' for _ in range (3)] for _ in range (3)]
    current_player='X'

    while True:
        print_board(board)

        if current_player=='X':
            row=int(input("Enter row no (1,2,3) :"))-1
            col=int(input("Enter col no (1,2,3) :"))-1

            if 0<=row<3 and 0<=col<3 and board[row][col]==' ':
                board[row][col]='X'
                if won(board,'X'):
                    print_board(board)
                    print("You won!")
                    break
                
                elif full(board):
                    print_board(board)
                    print("Game Tie!")
                    break
                current_player='O'
            else:
                print("Invalid move,Try again!")
        else:
            row,col=best_move(board)
            board[row][col]='O'
            if won(board,'O'):
                print_board(board)
                print("Computer Won!")
                break
            elif full(board):
                print_board(board)
                print("Game Tie!")
                break
            current_player='X'
while True:
    if __name__=="__main__":
        play()    