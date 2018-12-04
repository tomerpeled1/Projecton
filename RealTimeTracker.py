import cv2
import numpy as np

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
        min = conts_and_rects[0]
        minDis = dis(r_cent, min["center"])
        for c in conts_and_rects:
            if dis(c["center"], r_cent) < minDis:
                minDis = dis(c["center"], r_cent)
                min = c
        conts_and_rects.remove(min)
        d["centers"].append(min["center"])