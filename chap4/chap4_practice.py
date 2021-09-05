# 1th

# import sys
# import cv2
# import numpy as np

# src = cv2.imread("rose.bmp", cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print("Image load failed!")
#     sys.exit()

# #kernel = np.ones((3, 3), dtype = np.float64)/9

# #dst = cv2.filter2D(src, -1, kernel)

# dst = cv2.blur(src, (5, 5))

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 1th_2

# import cv2
# import numpy as np 
# import sys

# src = cv2.imread('rose.bmp', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print("Image load failed!")
#     sys.exit()

# cv2.imshow('src', src)

# for ksize in (3, 5, 7):
#     dst = cv2.blur(src, (ksize, ksize))
#     text = 'KernelSize : {} X {}'.format(ksize, ksize)
#     cv2.putText(dst, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 1, 255, 1, cv2.LINE_AA )
    
#     cv2.imshow('dst', dst)
#     cv2.waitKey()

# cv2.destroyAllWindows()


# 2th_1

# import cv2
# import numpy as np
# import sys

# src = cv2.imread('rose.bmp', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print("Image load failed!")
#     sys.exit()

# dst = cv2.GaussianBlur(src, (0, 0), 2)
# dst2 = cv2.blur(src, (7, 7))

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.imshow('dst2', dst2)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 2th_2

# import cv2
# import numpy as np
# import sys

# src = cv2.imread('rose.bmp', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print("Image load failed!")
#     sys.exit()

# cv2.imshow('src', src)

# for sigma in range(1, 6):
#     dst = cv2.GaussianBlur(src, (0, 0), sigma)
#     text = 'sigma : {}'.format(sigma)
#     cv2.putText(dst, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 1, 255, 1, cv2.LINE_AA)
#     cv2.imshow('dst', dst)
#     cv2.waitKey()

# cv2.destroyAllWindows()



# 3rd_1

# import sys
# import numpy as np
# import cv2

# src = cv2.imread('rose.bmp', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print('Imaage load failed!')
#     sys.exit()

# blr = cv2.GaussianBlur(src, (0, 0), 2)
# #dst = cv2.addWeighted(src, 2, blr, -1, 0)
# dst = np.clip(2.0*src - blr, 0, 255).astype(np.uint8)

# cv2.imshow('src', src)
# cv2.imshow('blr', blr)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 3rd_2

# import cv2
# import numpy as np
# import sys

# src = cv2.imread('rose.bmp', cv2.IMREAD_COLOR)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
# src_f = src_ycrcb[:, :, 0].astype(np.float32)
# blr = cv2.GaussianBlur(src_f, (0, 0), 2.0)
# src_ycrcb[:, :, 0] = np.clip(2*src_f - blr, 0, 255).astype(np.uint8)

# dst = cv2.cvtColor(src_ycrcb, cv2.COLOR_YCrCb2BGR)

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()



# 4th

# import sys
# import numpy as np
# import cv2

# src = cv2.imread('noise.bmp', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# dst = cv2.medianBlur(src, 3)

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()



# 5th

# import sys
# import numpy as np
# import cv2

# src= cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# dst = cv2.bilateralFilter(src, -1, 10, 5)

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()



# 6th

import sys
import cv2
import numpy as np

def cartoon_filter(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, (w//2, h//2))

    blr = cv2.bilateralFilter(img, -1, 20, 7)
    edge = 255 - cv2.Canny(img, 50, 120)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    dst = cv2.bitwise_and(blr, edge)
    dst = cv2.resize(dst, (w, h))

    return dst

def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray, (0, 0), 3)
    dst = cv2.divide(gray, blr, scale = 255)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Image open failed!')
    sys.exit()

mode = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if mode == 1:
        frame = cartoon_filter(frame)
    elif mode == 2:
        frame = pencil_sketch(frame)
    
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord(' '):
        mode +=1
        if mode ==3:
            mode = 0