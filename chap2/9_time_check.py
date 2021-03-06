import sys
import numpy as np
import cv2

img = cv2.imread('hongkong.jpg')
if img is None:
    print('Image loading is failed!')
    sys.exit()

tm = cv2.TickMeter()
tm.start()

edge = cv2.Canny(img, 50, 150)

tm.stop()
ms = tm.getTimeMilli()
print('Elaspes time : {}ms.'.format(ms))

cv2.imshow('img', img)
cv2.imshow('edge', edge)
cv2.waitKey()
cv2.destroyAllWindows()