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

import sys
import cv2
import numpy as np

src = cv2.imread('rose.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    sys.exit()

cv2.imshow('src', src)

for sigma in range(1, 6):
    dst = cv2.GaussianBlur(src, (0, 0), sigma)
    text = 'Sigma : {}'.format(sigma)
    cv2.putText(dst, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, 255, 1, cv2.LINE_AA)
    
    cv2.imshow('dst', dst)
    cv2.waitKey()

cv2.destroyAllWindows()