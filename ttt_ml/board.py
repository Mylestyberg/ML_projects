import numpy as np
from tqdm import tqdm




WIN_VALUE = 1.0  # type: float
DRAW_VALUE = 0.5  # type: float
LOSS_VALUE = 0.0  # type: float


class ttt_board():
    board = np.zeros((3,3))

    postions = {0: (0, 0), 1: (0, 1), 2: (0, 2),
                3: (1, 0), 4: (1, 1), 5: (1, 2),
                6: (2, 0), 7: (2, 1), 8: (2, 2)}

    # going to return new observation, reward, and whether player has won
    def make_move(self, action,aboard):
        x,y = self.postions.get(action)



        new_env = aboard.copy()

        new_env[x][y] = 1

        done,status = self.checkWinningStatus(new_env)

        if status == "CROSSES WIN":
           reward= -1  # type: float
        elif status == "NAUGHTS WIN":
            reward = 1
        elif status == "DRAW":
            reward = 0.5  # type: float
        else:
            reward = -0.2

        return new_env, done, reward



    def reset(self,board):
        board = np.zeros((3,3))
        return board



    def checkWinningStatus(self,aboard):
        potentials_winners = []
        potentials_winners.append(aboard[0][0] + aboard[0][1] + aboard[0][2])
        potentials_winners.append(aboard[1][0] + aboard[1][1] + aboard[1][2])
        potentials_winners.append(aboard[2][0] + aboard[2][1] + aboard[2][2])
        potentials_winners.append(aboard[0][0] + aboard[1][0] + aboard[2][0])
        potentials_winners.append(aboard[0][1] + aboard[1][1] + aboard[2][1])
        potentials_winners.append(aboard[0][2] + aboard[1][2] + aboard[2][2])
        potentials_winners.append(aboard[0][0] + aboard[1][1] + aboard[2][2])
        potentials_winners.append(aboard[0][2] + aboard[1][1] + aboard[2][0])

        return self.get_winner(potentials_winners,aboard)

    def get_winner(self, potential_winner,aboard):

        for addUp in potential_winner:
            if (addUp == 3):
                self.status = "NAUGHTS WIN"
                winner = True
                return winner, self.status
            elif (addUp == -3):
                self.status = "CROSSES WIN"
                winner = True
                return winner, self.status
            if self.check_if_table_full(aboard):
                self.status = "DRAW"
                winner = True
                return winner, self.status

        return False, "ON_GOING"



    def check_if_table_full(self,aboard):
        if ((abs(aboard[0][0]) + abs(aboard[0][1]) + abs(aboard[0][2]) +
             abs(aboard[1][0]) + abs(aboard[1][1]) + abs(aboard[1][2]) +
             abs(aboard[2][0]) + abs(aboard[2][1]) + abs(aboard[2][2])) == 9):
            return True
        else:
            return False

    def reshape_for_nn(self,aboard):
         reshape_board = np.zeros(27)
         count = 0
         while (count != 27):
            check = 0

            if count / 9>= 2:
                check = -1
            elif count / 9 >= 1:
                check = 1
            elif  count / 9 < 1:
                check= 0
            pointer = count
            for x in np.nditer(aboard):

                if x == check:
                    reshape_board[pointer] = 1

                pointer = pointer + 1

            count = count + 9

         return reshape_board.reshape(1,-1)



    def make_random_move(self,aboard):





       while True:
          rand_move = np.random.randint(0, 9)
          if not self.check_if_position(rand_move,aboard):
           x, y = self.postions.get(rand_move)
           aboard[x][y] = -1
           break


       done, status = self.checkWinningStatus(aboard)

       newboard = aboard.copy()
       reward = 0
       if status=="DRAW":
           reward=0
       else:
           reward= -1


       if done:
           return True,  reward,newboard

       return False,0, newboard








    def check_if_legal_move(self,position,aboard):
        if self.check_if_position(position,aboard) or (position>9 or position<0):
            return False

    def check_if_position(self,position,aboard):
        f = aboard[self.postions[position]]
        pieces = {1,-1}
        if position<0:
          return True
        elif  aboard[self.postions[position]] in pieces:
            return  True
        else:
           return False








if __name__ == '__main__':
    t = ttt_board()
    ttt_board.board[0][0] = -1
    print("debug")
    ttt_board().reshape_for_nn(ttt_board.board)
