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
    x = (box[0][0] + box[0][1]) / 2.0
    y = (box[1][0] + box[1][1]) / 2.0
    return (x, y)

def dis(x, y):
    return pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2)

def stupid_tracker(conts_and_rects, data):
    '''
    does the work of tracking - match contour, found by the detection, and a fruit, which is
    represented as a dictionary in data.
    at the end, conts_and_rects contains only the rects the did not match anything, aka the new fruits.
    '''
    for d in data:
        #if a fruit is falling
        if len(d["centers"]) > 1 and d["centers"][0][1] <= d["centers"][1][1]:
            print("ggggggggggggg")
            # todo do shit to remove contour

        if len(conts_and_rects.centers) > 0: #we found some fruits in the current pic
            x,y,w,h = d["window"]
            r = [(x,y), (x+w, y+h)]
            r_cent = center(r)
            n = len(conts_and_rects.centers)
            min = (0,0)
            minDis = 1000000000000000
            index = 0
            for i in range (0,n): # runs on all contours and finds the one with the smallest distance from
                                  # meanShift prediction
                if dis(conts_and_rects.centers[i], r_cent) < minDis:
                    minDis = dis(conts_and_rects.centers[i], r_cent)
                    min = conts_and_rects.centers[i]
                    index = i
            conts_and_rects.conts.pop(index) ##removes the contour so we won't look for it again.
            conts_and_rects.rects.pop(index)
            conts_and_rects.centers.pop(index)
            ## at the end, we add another center.
            d["centers"].append(min)

