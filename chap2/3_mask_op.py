import cv2
import sys

# src = cv2.imread('airplane.bmp', cv2.IMREAD_COLOR)
# mask = cv2.imread('mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
# dst = cv2.imread('field.bmp',cv2.IMREAD_COLOR)

# #cv2.copyTo(src, mask, dst)
# dst[mask>0] = src[mask>0]

# cv2.imshow('src', src)
# cv2.imshow('mask', mask)
# cv2.imshow('dst', dst)

# cv2.waitKey()
# cv2.destroyAllWindows()

#투명한 부분 있는 영상도 해보자

src = cv2.imread('opencv-logo-white.png', cv2.IMREAD_UNCHANGED)
mask = src[:, :, -1]
src = src[:, :, 0:3]
dst = cv2.imread('field.bmp',cv2.IMREAD_COLOR)

h,w = src.shape[:2]
crop = dst[10:h+10, 10:w+10]

cv2.copyTo(src,mask,crop)

cv2.imshow('src', src)
cv2.imshow('mask', mask)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()