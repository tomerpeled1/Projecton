import numpy as np
import cv2
import time



t = time.clock()

print(11111)

cap = cv2.VideoCapture(1)
print(cap.isOpened())
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# x = cv2.resize(x,(0,0), fx=0.3,fy= 0.3)
# y = cv2.imrea
# cv2.imshow("ttt", x)
# cv2.imwrite("tomer.png", x)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(time.clock()-t)