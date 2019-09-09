import numpy
import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, W, A, D,R ,ReleaseKey
import mss
import mss.tools
import carseour
import math


from google.cloud import vision



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


def detect_text(img):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()
    response = client.text_detection(image=img)
    texts = response.text_annotations
    return texts

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

    while True:

        #reward = get_reward()


        ##return current state
        if check ==30:
            print(3)


        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        screen_for_cnn = cv2.resize(screen, (80, 60))

        game = carseour.snapshot()
        f = carseour.models.GameInstance

        # print current speed of vehicle
        print(game.mSpeed * 2.24)
        if game.mCrashState == 0:
            crash = True
        else:
            crash = False

        reward(log_speed(game.mSpeed*2.24),crash)
        last_time = time.time()

        cv2.imshow('window', screen)



        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        return


def reset_env():
    PressKey(R)


def reward( log_speed, crashed):
    if  crashed:
        return -0.6
    elif log_speed < 1:
        return -0.04
    else:
        return log_speed


def log_speed(speed):
    x = (float(speed) / float((MAX_SPEED)) * 100.0) + 0.99
    base = 10
    return max(0.0, math.log(x, base) / 2.0)



def make_move(action,):
    key_dict = {}
    get_key =  key_dict[""]
    PressKey(get_key)
    new_state, done, reward = get_current_state()
    return  new_state, done, reward





if  __name__ == '__main__':
    get_current_state()


