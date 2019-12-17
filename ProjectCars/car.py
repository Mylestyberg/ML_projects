
import carseour
from directkeys import PressKey, W, A, D,R,S ,ReleaseKey
import time

game = carseour.live()




time.sleep(1)


PressKey(A)
ReleaseKey(D)
ReleaseKey(D)
print(game.mSteering)
ReleaseKey(A)


print(game.mSteering)
print(game.mSteering)
print(game.mSteering)


 ## need a function which tranlates speed to press time


