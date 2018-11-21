import numpy as np
import cv2
import time



t = time.perf_counter()

# print(11111)
real = cv2.imread("pic1.jpg")
real = cv2.resize(real,(0,0), fx=0.3,fy= 0.3)
back = cv2.imread("EmptyScreen.jpg")
back = cv2.resize(back,(0,0), fx=0.3,fy= 0.3)
subtract = np.zeros(real.shape, np.uint8)
height, width, no = real.shape

subtract = cv2.absdiff(real, back)
cv2.imshow("sub", subtract)
graysub = cv2.cvtColor(subtract, cv2.COLOR_BGR2GRAY)
ret, thresh_sub = cv2.threshold(graysub,3, 255, cv2.THRESH_TOZERO)

cv2.imshow("thresh_sub", thresh_sub)

# color_thresh_sub = cv2.cvtColor(thresh_sub, cv2.COLOR_GRAY2BGR)
# cv2.imshow("threshsub", thresh_sub)
morphed = cv2.morphologyEx(thresh_sub, cv2.MORPH_GRADIENT, None)
cv2.imshow("morphed", morphed)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, None)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, None)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, None)

cv2.imshow("closed", morphed)


# final = cv2.bitwise_and(real, thresh_sub)

# cv2.imshow("final", final)

# for i in range(height):
#     for j in range(width):
#         pix = [int(real[i,j][col]) - int(back[i,j][col]) for col in range(3)]
#         if not all([abs(col) < 8 for col in pix]):
#             pix = [real[i,j][col] for col in range(3)]
#         else:
#             pix = [0, 0, 0]
#         subtract[i,j] = pix
        # sub = real[i, j] - back[i,j]
        # if not np.any(sub == [0,0,0]):
        #     x = abs(sub)
        #     print(1)
        # if not np.any(abs(sub) < 30):
        #     sub += back[i,j]
        # else:
        #     sub = 0
        # subtract[i,j] = sub
# cv2.imshow("sub", subtract)
# cap = cv2.VideoCapture(1)
#print(cap.isOpened())
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


# x = cv2.resize(x,(0,0), fx=0.3,fy= 0.3)
# y = cv2.imrea
# cv2.imshow("ttt", x)
# cv2.imwrite("tomer.png", x)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(time.perf_counter()-t)
cv2.waitKey(0)
