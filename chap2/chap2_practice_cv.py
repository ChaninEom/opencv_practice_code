#1st

# import sys
# import cv2

# img1 = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('cat.bmp', cv2.IMREAD_COLOR)

# if img1 is None or img2 is None:
#     print("Image loading is failed")
#     sys.exit()

# print(img1.shape)
# print(type(img1))
# print(img1.dtype)


# if len(img1.shape) == 2:
#     print("img1 is grayscale")
# else:
#     print("img1 is colorscale")

# h,w = img1.shape[:2]
# print("{} {}".format(h,w))

# img1[:][:] = 0
# img2[:][:] = (255, 0, 255)

# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.waitKey()
# cv2.destroyAllWindows()



# 2nd

# import numpy as np
# import cv2

# img1 = np.empty((240, 320), dtype = np.uint8)
# img2 = np.zeros((240, 320), dtype = np.uint8)
# img3 = np.ones((240, 320), dtype = np.uint8)
# img4 = np.full((240, 320), 127, dtype = np.uint8)

# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)
# cv2.imshow('img4', img4)
# cv2.waitKey()
# cv2.destroyAllWindows()

# img1 = cv2.imread('HappyFish.jpg', cv2.IMREAD_COLOR)
# img2 = img1
# img3 = img1[50:150, 40:180].copy()
# print(img1.shape)

# img2[10:50, 20:50] = (255, 178, 35)

# cv2.imshow('img1',img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)
# cv2.waitKey()


# 3rd

# import cv2
# import sys

# src = cv2.imread('airplane.bmp', cv2.IMREAD_COLOR)
# mask = cv2.imread('mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
# dst = cv2.imread('field.bmp', cv2.IMREAD_COLOR)

# if src is None or mask is None or dst is None:
#     print("Image loading failed")
#     sys.exit()

# cv2.copyTo(src,mask,dst)
# #dst[mask>0] = src[mask>0]

# cv2.imshow('src', src)
# cv2.imshow('mask', mask)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

# src = cv2.imread('opencv-logo-white.png', cv2.IMREAD_UNCHANGED)
# mask = src[:, : , -1]
# src = src[:, :, :3]
# dst = cv2.imread('field.bmp', cv2.IMREAD_COLOR)

# h,w = src.shape[:2]
# crop = dst[20:20+h, 20:20+w]
# cv2.copyTo(src, mask, crop)

# cv2.imshow('src', src)
# cv2.imshow('mask', mask)
# cv2.imshow('dst',dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 4th
# import numpy as np
# import cv2

# img = np.full((400, 400, 3), 255, dtype = np.uint8)

# cv2.line(img, (50, 50), (200, 50), (0, 0, 255), 2)
# cv2.line(img, (50, 60), (150, 160), (0, 56, 150))

# cv2.rectangle(img, (150, 150, 100, 100), (0, 255, 0), 2, cv2.LINE_AA)
# cv2.rectangle(img, (160, 160, 80, 80), (255, 0, 0), -1)

# cv2.circle(img, (300, 100), 50, (26, 55, 132), 3, cv2.LINE_AA)
# cv2.circle(img, (300, 100), 20, (98, 55, 0), -1, cv2.LINE_AA)

# text ='Hello? Opencv' + cv2.__version__
# cv2.putText(img, text, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
# (0, 0, 255), 1, cv2.LINE_AA)

# cv2.imshow('img', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 5th

# import sys
# import cv2

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print('camera open failed')
#     sys.exit()

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(w, h)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# while(True):
#     ret, frame = cap.read()
    
#     if not ret:
#         break

#     edge = cv2.Canny(frame, 50, 150)

#     cv2.imshow('edge', edge)
#     cv2.imshow('frame', frame)
    
#     if cv2.waitKey(20) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# import sys
# import cv2

# cap = cv2.VideoCapture('video1.mp4')
# if not cap.isOpened():
#     print('video open failed!')

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(w, h)

# while(True):
#     ret, frame = cap.read()
#     if not ret:
#         break
#     edge = cv2.Canny(frame, 50, 150)

#     cv2.imshow('edge', edge)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(20) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


# 6th

# import cv2
# import sys

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print('camera open failed!')
#     sys.exit()

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(w, h)

# fps = round(cap.get(cv2.CAP_PROP_FPS))
# delay = round(1000/fps)

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output.avi', fourcc, fps, (w,h))

# if not out.isOpened():
#     print('File open failed!')
#     cap.release()
#     sys.exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     edge = cv2.Canny(frame, 50, 150)
#     edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

#     out.write(edge_color)
#     cv2.imshow('frame', frame)
#     cv2.imshow('edge', edge_color)

#     if cv2.waitKey(delay) == 27:
#         break

# out.release()
# cap.release()
# cv2.destroyAllWindows()


# 7th

# import sys
# import cv2
# import numpy as np

# img = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)

# if img is None:
#     print("imgae load failed")
#     sys.exit()

# cv2.namedWindow('img')
# cv2.imshow('img', img)

# while True:
#     key = cv2.waitKey()
    
#     if key == 27:
#         break
#     elif key == ord('i') or ord('I'):
#         img = ~img
#         cv2.imshow('img', img)

# cv2.destroyAllWindows()

# 8th

# import sys
# import numpy as np
# import cv2

# oldx = oldy = -1
# def on_mouse(event, x, y, flags, param):
#     global img, oldx, oldy
#     if event == cv2.EVENT_LBUTTONDOWN:
#         oldx, oldy = x, y
#         print('EVENT_LBUTTONDOWN : {}, {}'.format(x, y))
#     elif event == cv2.EVENT_LBUTTONUP:
#         print('EVENT_LBUTTONUO : {}, {}'.format(x, y))
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if flags & cv2.EVENT_LBUTTONDOWN:
#             cv2.line(img, (oldx, oldy), (x, y), (0, 0, 255), 3, cv2.LINE_AA)
#             cv2.imshow('img', img)
#             oldx, oldy = x, y

# img = np.ones((480, 640, 3), dtype = np.uint8)*255

# if img is None:
#     print('image make failed!')
#     sys.exit()

# cv2.namedWindow('img')
# cv2.setMouseCallback('img', on_mouse)
# cv2.imshow('img', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 8th 2nd

# import sys
# import cv2
# import numpy as np

# def on_trackbar(pos):
#     global img
#     level = pos*16
#     # if level >=255:
#     #     level = 255
#     level = np.clip(level, 0, 255)
#     img[:, :] = level
#     cv2.imshow('img', img)

# img = np.zeros((480, 640), np.uint8)
# if img is None:
#     print('image load failed')
#     sys.exit()

# cv2.namedWindow('img')
# cv2.createTrackbar('level', 'img', 0, 16, on_trackbar)
# cv2.imshow('img', img)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 9th

# import sys
# import cv2
# import numpy as np

# img = cv2.imread('hongkong.jpg')
# if img is None:
#     print('image load failed')
#     sys.exit()

# tm = cv2.TickMeter()

# for i in range(1, 20):
#     tm.start()
#     edge = cv2.Canny(img, 50, 150)
#     tm.stop()
#     ms = tm.getTimeMilli()
#     print('Time for Canny is {}ms', ms)
#     tm.reset()

# cv2.imshow('img', img)
# cv2.imshow('edge', edge)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 10th

# import cv2
# import numpy as np
# import sys


# cap1 = cv2.VideoCapture('video1.mp4')
# cap2 = cv2.VideoCapture('video2.mp4')

# if not cap1.isOpened() or not cap2.isOpened():
#     print('Video open failed!')
#     sys.exit()

# fps = round(cap1.get(cv2.CAP_PROP_FPS))
# delay = round(1000/fps)

# w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(w, h)
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output.avi', fourcc, fps, (w,h))

# while True:
#     ret1, frame1 = cap1.read()
#     if not ret1:
#         break

#     out.write(frame1)
#     cv2.imshow('frame', frame1)
#     cv2.waitKey(delay)


# while True:
#     ret2, frame2 = cap2.read()
#     if not ret2:
#         break
#     out.write(frame2)
#     cv2.imshow('fram', frame2)
#     cv2.waitKey(delay)

# cap1.release()
# cap2.release()
# cv2.destroyAllWindows()


# 11th

import cv2
import sys
import numpy as np

cap1 = cv2.VideoCapture('video1.mp4')
cap2 = cv2.VideoCapture('video2.mp4')
if not cap1.isOpened() or not cap2.isOpened():
    print('video open failed!')
    sys.exit()

frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(cap1.get(cv2.CAP_PROP_FPS))
delay = int(1000/fps)

print('frame cnt1 = {}'.format(frame_cnt1))
print('frame_cnt2 = {}'.format(frame_cnt2))
print('fps = {}'.format(fps))
effect_frame = int(fps*2)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

for i in range(frame_cnt1 - effect_frame):
    ret1, frame1 = cap1.read()
    if not ret1 :
        break
    out.write(frame1)
    cv2.imshow('frame', frame1)
    cv2.waitKey(delay)

for i in range(effect_frame):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    dx = int(w*i/effect_frame)
    frame = np.zeros((h, w , 3), np.uint8)

    frame[:, 0:dx] = frame2[:, 0:dx]
    frame[:, dx:w] = frame1[:, dx:w]
    out.write(frame)
    cv2.imshow('frame', frame)
    cv2.waitKey(delay)

for i in range(effect_frame, frame_cnt2):
    ret2, frame2 = cap2.read()
    if not ret2:
        break
    out.write(frame2)
    cv2.imshow('frame', frame2)
    cv2.waitKey(delay)

out.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()