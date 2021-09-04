import sys
import cv2

img1 = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('cat.bmp', cv2.IMREAD_COLOR)

if img1 is None or img2 is None:
    print("Image load failed!!")
    sys.exit()

print('type(img1) : ', type(img1))
print('img1.shape : ', img1.shape)
print('img2.shape : ', img2.shape)
print('img1.dtype : ', img1.dtype)
print('img2.dtype : ', img2.dtype)

if len(img1) == 2:
    print("grayscale!")
else:
    print("colorscale!")

h,w = img1.shape[:2]
# print("w x h = {} x {}".format(w, h))

# x = 20
# y = 10
# p1 = img1[y,x]
# print(p1)

# p2 = img2[y,x]
# print(p2)

# 이런 형태 개느림 주의!
# for y in range(h):
#     for x in range (w):
#         img1[y,x] = 0
#         img2[y,x] = (0,255,255)

img1[:][:] =0
img2[:][:] = (0, 255, 255)

cv2.imshow('img1', img1)
cv2.imshow('img2',img2)
cv2.waitKey()