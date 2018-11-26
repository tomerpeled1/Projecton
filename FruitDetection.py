import numpy as np
import cv2
import time
from PIL import Image

def fruit_detection(frame, background):
    t = time.perf_counter()

    real = frame
    back = background


    real_h = cv2.cvtColor(real, cv2.COLOR_BGR2HSV)[:,:,0]
    back_h = cv2.cvtColor(back, cv2.COLOR_BGR2HSV)[:,:,0]

    subtract = cv2.absdiff(real, back)
    graysub = cv2.cvtColor(subtract, cv2.COLOR_BGR2GRAY)
    ret1, sub_mask = cv2.threshold(graysub, 45, 255, cv2.THRESH_BINARY)

    subtract_h = cv2.absdiff(real_h, back_h)
    opened_h = cv2.morphologyEx(subtract_h, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    ret2, sub_h_mask = cv2.threshold(opened_h, 10, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(sub_mask, sub_h_mask)

    # ret, thresh_sub = cv2.threshold(norm_sub, 45, 255, cv2.THRESH_TOZERO)
    # cv2.imshow("c", thresh_sub)
    # _ , masked = cv2.bitwise_and(real,real, mask=mask)
    # cv2.imshow("d", masked)
    # hsvsub = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    # morphed = cv2.morphologyEx(thresh_sub, cv2.MORPH_GRADIENT, None)


    # h = hsvsub[:,:,0]
    # cv2.imshow("e", h)
    # h = cv2.blur(h, (5,5))
    # cv2.imshow("blur", h)
    # h_array = cv2.normalize(h, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    # cv2.imshow("f", h_array)

    # morphed = cv2.morphologyEx(masked, cv2.MORPH_GRADIENT, None)
    # cv2.imshow("g", morphed)

    masked = cv2.bitwise_and(real, real, mask=mask)
    gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    im2, cont, hier = cv2.findContours(gray_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = [c for c in cont if cv2.contourArea(c) > 4000]
    print(time.perf_counter()-t)

    return cont

if __name__ == "__main__":
    frame = cv2.imread("pic4.jpg")
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    (height, width, depth) = frame.shape
    back = cv2.imread("pic2.jpg")
    back = cv2.resize(back, None, fx=0.3, fy=0.3)

    cont = fruit_detection(frame, back)
    cv2.drawContours(frame, cont, -1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()