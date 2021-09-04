import numpy as np
import cv2
import sys

def on_trackbar(pos):
    global img
    print(pos)
    level = pos*16
    # if level >=255:
    #     level = 255
    level = np.clip(level, 0, 255)
    img[:,:] = level
    cv2.imshow('image', img)

img = np.zeros((480, 640), np.uint8)
if img is None:
    print("Image making is failed!")
    sys.exit()

cv2.namedWindow('image')

cv2.createTrackbar('level', 'image', 0, 16, on_trackbar)

cv2.imshow('image', img)
cv2.waitKey()

cv2.destroyAllWindows()