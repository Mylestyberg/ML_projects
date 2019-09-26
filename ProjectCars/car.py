
import carseour
import time

game = carseour.live()

while True:
    print("Speed: " + str(round(game.mSpeed, 1)) + " m/s")
    print(game.mSteering)
    print(game.mBrake)
    print(game.mThrottle)
    print(game.mCurrentSector1Time)
    print(game.mCurrentSector2Time)
    print(game.mCurrentSector3Time)
    time.sleep(0.5)
