import numpy as np
from PIL import ImageGrab
import cv2
import time
from numpy import ones,vstack
from numpy.linalg import lstsq
from directkeys import PressKey, W, A, S, D
from statistics import mean




def process_img(image):

    original_image = image
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    mask_white = cv2.inRange(gray_image, 200, 255)

    mask_yw_image = cv2.bitwise_and(gray_image, mask_white)

    processed_img = mask_yw_image

    kernel_size = 5
    gauss_gray = cv2.GaussianBlur(mask_yw_image, kernel_size)

    low_threshold = 50
    high_threshold = 150
    canny_edges = cv2.Canny(gauss_gray, low_threshold, high_threshold)


    return processed_img, original_image


def main():
    last_time = time.time()
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        print('Frame took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        new_screen, original_image = process_img(screen)
        cv2.imshow('window', new_screen)
        cv2.imshow('window2', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


main()