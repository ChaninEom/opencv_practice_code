import sys
import numpy as np
import cv2

cap1 = cv2.VideoCapture('woman.mp4')
if not cap1.isOpened():
    print('Video open failed!')
    sys.exit()

cap2 = cv2.VideoCapture('raining.mp4')
if not cap2.isOpened():
    print('V(ideo2 open failed!')
    sys.exit()

w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_Cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

print('w x h : {} x {}'.format(w, h))
print('frame_cnt1 :', frame_cnt1)
print('frame_cnt2 :', frame_Cnt2)

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
        mask = cv2.inRange(hsv, (50, 150, 0), (80, 255, 255))
        cv2.copyTo(frame2, mask, frame1) #frame2에서 mask가 흰색인 부분만 frame1으로 복사하라는 뜻
    
    out.write(frame1)
    cv2.imshow('frame', frame1)
    key = cv2.waitKey(delay-20)

    if key == ord(' '):
        do_composit = not do_composit
    elif key == 27:
        break

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
