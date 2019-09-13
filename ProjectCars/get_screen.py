import numpy
import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, W, A, D,R,S ,ReleaseKey
import mss
import mss.tools
import carseour
import math


from stopwatch import Stopwatch

stopwatch = Stopwatch()




def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


actions = {0: straight, 1: right, 2: left,3:slow_ya_roll}





import os



MAX_SPEED = 180
check =0



def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)


def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)



def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)



def get_screem():

 with mss.mss() as sct:
     with mss.mss() as sct:
         # Part of the screen to capture
         monitor = {"top": 0, "left": 40, "width": 800, "height": 640}

         while "Screen capturing":
             last_time = time.time()

             # Get raw pixels from the screen, save it to a Numpy array
             img = numpy.array(sct.grab(monitor))

             # Display the picture
             cv2.imshow("OpenCV/Numpy normal", img)


             print("fps: {}".format(1 / (time.time() - last_time)))

             # Press "q" to quit
             if cv2.waitKey(25) & 0xFF == ord("q"):
                 cv2.destroyAllWindows()

                 break

 global check
 check =0
def  get_current_state():
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        screen_for_cnn_state = cv2.resize(screen, (80, 60))

        game = carseour.snapshot()

        # print current speed of vehicle

        if game.mCrashState == 1:
            crash = True
        else:
            crash = False

        penalty = 0

        if game.mSpeed < 1.3:
           stopwatch.start()
           if stopwatch.duration > 25:
             reset_env()
             stopwatch.reset()





        reward_ = reward(log_speed(game.mSpeed*2.24),crash,game.mSpeed*2.24)

        return screen_for_cnn_state, reward_


def reset_env():
    PressKey(R)
    time.sleep(0.5)
    ReleaseKey(R)


def reward( log_speed, crashed,speed):


    if  crashed  :
        return -0.7
    elif speed < 6:
         return -0.04
    else:
        return log_speed



def log_speed(speed):
    x = (float(speed) / float((MAX_SPEED)) * 100.0) + 0.99
    base = 10
    return max(0.0, math.log(x, base) / 2.0)



def make_move(action,):
    get_move = actions[action]
    get_move()
    new_state,  reward = get_current_state()
    return  new_state, reward


def start_screen(): ## need this done
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    screen_for_cnn_state = cv2.resize(screen, (80, 60))
    return screen_for_cnn_state





if  __name__ == '__main__':
    get_current_state()


