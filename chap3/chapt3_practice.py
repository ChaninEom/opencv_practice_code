# 1st

# import sys
# import numpy as np
# import cv2

# src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)
# dst = cv2.add(src, 100)

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import sys
# import cv2
# import numpy as np

# src = cv2.imread('lenna.bmp', cv2.IMREAD_COLOR)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# dst = cv2.add(src, (100, 100, 100, 0))

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 2nd

# import sys
# import cv2
# import matplotlib.pyplot as plt

# src1 = cv2.imread('lenna256.bmp', cv2.IMREAD_GRAYSCALE)
# src2 = cv2.imread('square.bmp', cv2.IMREAD_GRAYSCALE)

# if src1 is None or src2 is None:
#     print('Image load failed!')
#     sys.exit()

# dst1 = cv2.add(src1, src2)
# dst2 = cv2.addWeighted(src1, 0.5, src2, 0.5, 0.0)
# dst3 = cv2.subtract(src1, src2)
# dst4 = cv2.absdiff(src1, src2)

# plt.subplot(231), plt.axis('off'), plt.imshow(src1, 'gray'), plt.title('src1')
# plt.subplot(232), plt.axis('off'), plt.imshow(src2, 'gray'), plt.title('src2')
# plt.subplot(233), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title('add')
# plt.subplot(234), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('addWeighted')
# plt.subplot(235), plt.axis('off'), plt.imshow(dst3, 'gray'), plt.title('subtract')
# plt.subplot(236), plt.axis('off'), plt.imshow(dst4, 'gray'), plt.title('absdiff')

# plt.show()


# 3rd

# import numpy as np
# import sys
# import cv2

# src = cv2.imread('candies.png', cv2.IMREAD_COLOR)

# if src is None:
#     print('Imaga load failed!')
#     sys.eixt()

# print('src.shape : ', src.shape)
# print('src.dtye : ', src.dtype)

# planes = cv2.split(src)

# cv2.imshow('src', src)

# cv2.imshow('planes[0]', planes[0])
# cv2.imshow('planes[1]', planes[1])
# cv2.imshow('planes[2]', planes[2])

# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import sys

# src = cv2.imread('candies.png', cv2.IMREAD_COLOR)

# if src is None:
#     print('Image laod failed!')
#     sys.exit()

# print('src.shape :', src.shape)
# print('src.dtype :', src.dtype)

# src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# planes = cv2.split(src_hsv)

# cv2.imshow('src', src)

# cv2.imshow('planes[0]', planes[0])
# cv2.imshow('planes[1]', planes[1])
# cv2.imshow('planes[2]', planes[2])

# cv2.waitKey()
# cv2.destroyAllWindows()


# 4th_1

# import sys
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# hist = cv2.calcHist([src], [0], None, [256], [0, 255])

# cv2.imshow('src', src)
# cv2.waitKey(1)

# plt.plot(hist, 'gray')
# plt.show()

# import sys
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# src = cv2.imread('lenna.bmp', cv2.IMREAD_COLOR)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# colors = ['b', 'g', 'r']
# bgr_planes = cv2.split(src)

# for (p, c) in zip(bgr_planes, colors):
#     hist = cv2.calcHist([p], [0], None, [256], [0, 256])
#     plt.plot(hist, c)

# cv2.imshow('src', src)
# cv2.waitKey(1)
# plt.show()

# 4th_2

# import sys
# import numpy as np
# import cv2

# def getGrayHist(hist):
#     histImg = np.full((100, 256), 255, dtype = np.uint8)
#     if histImg is None:
#         print('Image make faield!')
#         sys.eixt()
    
#     histmax = np.max(hist)

#     for x in range(256):
#         pt1 = (x, 100)
#         pt2 = (x, 100-int(hist[x, 0]*100/histmax))
#         cv2.line(histImg, pt1, pt2, 0)
    
#     return histImg

# src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)
# if src is None:
#     print('Image load failed!')
#     sys.exit()

# hist = cv2.calcHist([src], [0], None, [256], [0, 256])
# histImg = getGrayHist(hist)

# cv2.imshow('src', src)
# cv2.imshow('histImg', histImg)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 5th_1

# import sys
# import cv2
# import numpy as np

# src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)
# if src is None:
#     print('Image load failed!')
#     sys.exit()

# alpha = 1.0
# print(src.dtype)

# dst = np.clip((1+alpha)*src-128*alpha, 0, 255).astype(np.uint8)
# print(dst.dtype)

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import sys

# def getHistImg(hist):
#     histImg = np.full((100,256), 255, dtype = np.uint8)
#     if histImg is None:
#         print('Image make failed!')
#         sys.exit()
#     histmax = np.max(hist)
#     for x in range(256):
#         pt1 = (x, 100)
#         pt2 = (x, 100-int(hist[x, 0]*100/histmax))
#         cv2.line(histImg, pt1, pt2, 0)
#     return histImg

# src = cv2.imread('Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# gmax = np.max(src)
# gmin = np.min(src)

# dst = np.clip((src-gmin)*255./(gmax-gmin), 0 , 255).astype(np.uint8)
# print('dst.dtype : {}'.format(dst.dtype))

# src_hist = cv2.calcHist([src], [0], None, [256], [0, 256])
# dst_hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

# src_histImg = getHistImg(src_hist)
# dst_histImg = getHistImg(dst_hist)

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.imshow('src_hist', src_histImg)
# cv2.imshow('dst_hist', dst_histImg)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 6th

# import cv2
# import numpy as np
# import sys

# src = cv2.imread('Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print("Image load failed!")
#     sys.exit()

# dst = cv2.equalizeHist(src)

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# import sys
# import numpy as np

# src = cv2.imread('field.bmp')

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
# ycrcb_planes = cv2.split(src_ycrcb)

# ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
# dst_ycrcb = cv2.merge(ycrcb_planes)

# dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 7th

# import sys
# import numpy as np
# import cv2

# #src = cv2.imread('candies.png')
# src = cv2.imread('candies2.png')

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# dst1 = cv2.inRange(src, (0, 128, 0), (100, 256, 100))
# dst2 = cv2.inRange(src_hsv, (50, 150, 0), (80, 255, 255))

# cv2.imshow('src', src)
# cv2.imshow('dst1', dst1)
# cv2.imshow('dst2', dst2)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 7th_2

# import sys
# import numpy as np
# import cv2

# def on_trackbar(pos):
#     hmax = cv2.getTrackbarPos('H_max', 'dst')
#     hmin = cv2.getTrackbarPos('H_min', 'dst')
#     dst = cv2.inRange(src_hsv, (hmin, 150, 0), (hmax, 255, 255))
#     cv2.imshow('dst', dst)

# src = cv2.imread('candies.png', cv2.IMREAD_COLOR)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# hsv_planes = cv2.split(src_hsv)

# hsv_planes[0] = cv2.equalizeHist(hsv_planes[0])
# dst_hsv = cv2.merge(hsv_planes)

# dst = cv2.cvtColor(dst_hsv, cv2.COLOR_HSV2BGR)

# cv2.imshow('src', src)
# cv2.namedWindow('dst')

# cv2.createTrackbar('H_min', 'dst', 50, 179, on_trackbar)
# cv2.createTrackbar('H_max', 'dst', 80, 179, on_trackbar)
# on_trackbar(0)

# cv2.waitKey()



# 8th_1

# import sys
# import cv2

# src = cv2.imread('cropland.png')

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# x, y, w, h = cv2.selectROI(src)

# src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
# crop = src_ycrcb[y:y+h, x:x+w]

# channels = [1, 2]
# Cr_bins = 128
# Cb_bins = 128
# hist_size = [Cr_bins, Cb_bins]
# Cr_range = [0, 256]
# Cb_range = [0, 256]
# ranges = Cr_range + Cb_range

# hist = cv2.calcHist([crop], channels, None, hist_size, ranges)
# hist_norm = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1)
# dst = cv2.copyTo(src, backproj)

# cv2.imshow('src', src)
# cv2.imshow('hist_norm', hist_norm)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2. destroyAllWindows()


# 8tn_2

# import cv2
# import sys

# ref = cv2.imread('kids1.png', cv2.IMREAD_COLOR)
# mask = cv2.imread('kids1_mask.bmp', cv2.IMREAD_GRAYSCALE)

# if ref is None or mask is None:
#     print('Image load failed!')
#     sys.exit()

# ref_ycrcb = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

# channels = [1, 2]
# ranges = [0, 256, 0, 256]
# bins = [128, 128]
# hist = cv2.calcHist([ref_ycrcb], channels, mask, bins, ranges)
# hist_norm = cv2.normalize(cv2.log(hist+1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# src = cv2.imread('kids2.png', cv2.IMREAD_COLOR)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
# backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1)

# cv2.imshow('src', src)
# cv2.imshow('hist_norm', hist_norm)
# cv2.imshow('backproj', backproj)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 9th

import sys
import cv2

cap1 = cv2.VideoCapture('woman.mp4')
if not cap1.isOpened():
    print('Cap1 open failed!')
    sys.exit()

cap2 = cv2.VideoCapture('raining.mp4')
if not cap2.isOpened():
    print('Cap2 open failed!')
    sys.exit()

w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

print('w x h : {} x {}'.format(w, h))
print('frame_cnt1 :', frame_cnt1)
print('frame_cnt2 :', frame_cnt2)

fps = cap1.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

do_composit = False

while True:
    ret1, frame1 = cap1.read()
    if not ret1:
        break

    if do_composit:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (50, 150, 0), (80,255, 255))
        cv2.copyTo(frame2, mask, frame1)
    
    out.write(frame1)
    cv2.imshow('frame', frame1)
    key = cv2.waitKey(delay-10)

    if key == ord(' '):
        do_composit = not do_composit

    elif key == 27:
        break

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()