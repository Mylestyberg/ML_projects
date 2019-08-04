import numpy as np
class ttt_board():
    board = np.zeros((3,3))

    postions = {0: (0, 0), 1: (0, 1), 2: (0, 2),
                3: (1, 0), 4: (1, 1), 5: (1, 2),
                6: (2, 0), 7: (2, 1), 8: (2, 2)}

    # going to return new observation, reward, and whether player has won
    def make_move(self, action):
        input = self.postions.get(action)



    #check winner
    def check_win_state(self):


    def reshape_for_nn(self,board):
       reshape_board = np.zeros(27)
       count = 27
       while (count != 0):
        check = 0

        if count % 9 == 3:
            check = 1
        elif count % 9 == 2:
            check = -1
        elif  count % 9 == 1:
            check=0

        for x in np.nditer(self.board):
            if x == check:
                reshape_board[check] = x








if __name__ == '__main__':
    t = ttt_board()
    ttt_board.board[0][0] = -1
    print("debug")
    ttt_board().reshape_for_nn(ttt_board.board)
