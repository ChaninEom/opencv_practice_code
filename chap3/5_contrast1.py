import sys
import cv2
import numpy as np


src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('image load failed!')
    sys.exit()

alpha = 1.0
dst = np.clip((1+alpha)*src-128*alpha, 0, 255).astype(np.uint8)


cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()