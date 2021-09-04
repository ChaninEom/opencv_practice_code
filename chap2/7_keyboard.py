import sys
import numpy as np
import cv2

img = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("image loading is failed")
    sys.exit()

cv2.namedWindow('image')
cv2.imshow('image', img)

while True:
    key = cv2.waitKey()
    if key == 27:
        break
    elif key == ord('i') or ord('I'):
        img = ~img
        cv2.imshow('image', img) 
cv2.destroyAllWindows()