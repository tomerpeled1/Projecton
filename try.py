import numpy as np
import cv2
import time


def fruit_detection(frame, background):
    t = time.perf_counter()

    real = frame
    real = cv2.resize(real,(0,0), fx=0.3,fy= 0.3)
    cv2.imshow("real", real)
    cv2.waitKey(0)
    back = background
    back = cv2.resize(back,(0,0), fx=0.3,fy= 0.3)

    subtract = cv2.absdiff(real, back)
    graysub = cv2.cvtColor(subtract, cv2.COLOR_BGR2GRAY)
    ret, thresh_sub = cv2.threshold(graysub,40, 255, cv2.THRESH_TOZERO)

    morphed = cv2.morphologyEx(thresh_sub, cv2.MORPH_GRADIENT, None)
    im2, cont, hier = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    print(time.perf_counter()-t)

    return  cont

