# import sys
# import numpy as np
# import cv2

# src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)
# dst = cv2.add(src, 100)
# #dst = np.clip(src +100., 0, 255).astype(np.uint8)


# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()

# cv2.destroyAllWindows()

import sys
import cv2
import numpy as np

src = cv2.imread('lenna.bmp', cv2.IMREAD_COLOR)

if src is None:
    print('Image load failed!')
    sys.exit()

dst = cv2.add(src, (100, 100, 100, 0))

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()