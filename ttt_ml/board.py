import numpy as np
from tqdm import tqdm


EPISODES = 20_000


WIN_VALUE = 1.0  # type: float
DRAW_VALUE = 0.5  # type: float
LOSS_VALUE = 0.0  # type: float

learning_rate = 0.9
value_discount = 0.95
class ttt_board():
    board = np.zeros((3,3))

    postions = {0: (0, 0), 1: (0, 1), 2: (0, 2),
                3: (1, 0), 4: (1, 1), 5: (1, 2),
                6: (2, 0), 7: (2, 1), 8: (2, 2)}

    # going to return new observation, reward, and whether player has won
    def make_move(self, action):
        x,y = self.postions.get(action)

        self.board[x][y] = 1

        new_env = self.board.copy()

        done,status = self.checkWinningStatus()

        if status == "CROSSES WIN":
           reward= 1  # type: float
        elif status == "NAUGHTS WIN":
            reward = -1
        elif status == "DRAW":
            reward = 0.5  # type: float
        else:
            reward = 0

        return new_env, done, reward







    def checkWinningStatus(self):
        potentials_winners = []
        potentials_winners.append(self.board[0][0] + self.board[0][1] + self.board[0][2])
        potentials_winners.append(self.board[1][0] + self.board[1][1] + self.board[1][2])
        potentials_winners.append(self.board[2][0] + self.board[2][1] + self.board[2][2])
        potentials_winners.append(self.board[0][0] + self.board[1][0] + self.board[2][0])
        potentials_winners.append(self.board[0][1] + self.board[1][1] + self.board[2][1])
        potentials_winners.append(self.board[0][2] + self.board[1][2] + self.board[2][2])
        potentials_winners.append(self.board[0][0] + self.board[1][1] + self.board[2][2])
        potentials_winners.append(self.board[0][2] + self.board[1][1] + self.board[2][0])

        return self.get_winner(potentials_winners)

    def get_winner(self, potential_winner):
        for addUp in potential_winner:
            if (addUp == 3):
                self.status = "NAUGHTS WIN"
                return True, self.status
            elif (addUp == -3):
                self.status = "CROSSES WIN"
                return True,self.status
            elif self.check_if_table_full():
                self.status = "DRAW"
                return True, self.status
            else:
                return False, "ON_GOING"

    def check_if_table_full(self):
        if ((abs(self.board[0][0]) + abs(self.board[0][1]) + abs(self.board[0][2]) +
             abs(self.board[1][0]) + abs(self.board[1][1]) + abs(self.board[1][2]) +
             abs(self.board[2][0]) + abs(self.board[2][1]) + abs(self.board[2][2])) == 9):
            return True
        else:
            return False

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



for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    done = False
    #while not done:




if __name__ == '__main__':
    t = ttt_board()
    ttt_board.board[0][0] = -1
    print("debug")
    ttt_board().reshape_for_nn(ttt_board.board)
