import numpy as np
import cv2
import sys

ref = cv2.imread('kids1.png', cv2.IMREAD_COLOR)
mask = cv2.imread('kids1_mask.bmp', cv2.IMREAD_GRAYSCALE)

if ref is None or mask is None:
    print('Image load faield!')
    sys.exit()

ref_ycrcb = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

channels = [1, 2]
range = [0, 256, 0, 256]
hist = cv2.calcHist([ref_ycrcb], channels, mask, [128, 128], range)
hist_Norm = cv2.normalize(cv2.log(hist+1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

src = cv2.imread('kids2.png', cv2.IMREAD_COLOR)

if src is None:
    print('Image src load failed!')
    sys.exit()

src_ycrcb =  cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
backporj = cv2.calcBackProject([src_ycrcb], channels, hist, range, 1)    

cv2.imshow('src', src)
cv2.imshow('hist_norm', hist_Norm)
cv2.imshow('backproj', backporj)
cv2.waitKey()
cv2.destroyAllWindows()