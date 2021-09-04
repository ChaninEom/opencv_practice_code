# import numpy as np
# import sys
# import cv2

# src = cv2.imread('candies.png', cv2.IMREAD_COLOR)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# print('src.shape : ', src.shape)
# print('src.dtpye : ', src.dtype)

# planes = cv2.split(src)

# cv2.imshow('src', src)
 
# cv2.imshow('planes[0]', planes[0])
# cv2.imshow('planes[1]', planes[1])
# cv2.imshow('planes[2]', planes[2])

# cv2.waitKey()

# cv2.destroyAllWindows()

import numpy as np
import sys
import cv2

src = cv2.imread('candies.png', cv2.IMREAD_COLOR)

if src is None:
    print('Image load failed!')
    sys.exit()

print('src.shape : ', src.shape)
print('src.dtpye : ', src.dtype)

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
planes = cv2.split(src_hsv)

cv2.imshow('src', src)
 
cv2.imshow('planes[0] : hue', planes[0]) #색상
cv2.imshow('planes[1] : saturation', planes[1]) #채도
cv2.imshow('planes[2] :', planes[2]) #명도

cv2.waitKey()

cv2.destroyAllWindows()