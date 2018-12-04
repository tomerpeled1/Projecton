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

def stupid_tracker(conts, data):
    conts_centers = []
    for c in conts:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        conts_centers.append((cX, cY))
    for d in data:
        r = d["window"]
        r_cent = center(r)
        min = conts_centers[0]
        minDis = dis(r_cent, min)
        for c in conts_centers:
            if dis(c, r_cent) < minDis:
                minDis = dis(c, r_cent)
                min = c
        conts_centers.remove(min)
        d["centers"].append(min)