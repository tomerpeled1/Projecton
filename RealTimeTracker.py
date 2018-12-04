import cv2
import numpy as np

CONT = 0
RECT = 1
CENTER = 2


def center(box):
    '''
    returns center of a box.
    '''
    # return a x and y position
    x = box[0][0] + box[0][1] / 2.0
    y = box[1][0] + box[1][1] / 2.0
    return (x, y)

def dis(x, y):
    return pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2)

def stupid_tracker(conts_and_rects, data):
    for d in data:
        r = d["window"]
        r_cent = center(r)
        n = len(conts_and_rects[CENTER])
        min = conts_and_rects[CENTER][0]
        minDis = dis(r_cent, min)
        index = 0
        for i in range (1,n):
            if dis(conts_and_rects[CENTER][i], r_cent) < minDis:
                minDis = dis(conts_and_rects[CENTER][i], r_cent)
                min = conts_and_rects[CENTER][i]
                index = i
        del conts_and_rects[:][index]
        d["centers"].append(min)