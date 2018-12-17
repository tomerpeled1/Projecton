import cv2
import numpy as np
from Fruit import Fruit

NUM_OF_FRAMES = 5


def center(box):
    '''
    returns center of a box.
    '''
    # return a x and y position
    x = (box[0][0] + box[1][0]) / 2.0
    y = (box[0][1] + box[1][1]) / 2.0
    return (x, y)

def dis(x, y):
    return pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2)

def update_falling(fruit):
    assert len(fruit.centers) > 1
    if fruit.centers[0][1] <= fruit.centers[1][1]:
        fruit.is_falling = True

def track_object(detection_results, fruit):
    if len(detection_results.centers) > 0:
        x, y, w, h = fruit.track_window
        r = [(x, y), (x+w, y+h)]
        r_cent = center(r)
        n = len(detection_results.centers)
        min = detection_results.centers[0]
        min_dis = dis(r_cent, min)
        index = 0
        for i in range(1,n):
            if dis(detection_results.centers[i], r_cent) < min_dis:
                min = detection_results.centers[i]
                min_dis = dis(r_cent, min)
                index = i
        detection_results.conts.pop(index)  ##removes the contour so we won't look for it again.
        fruit.track_window = box2window(detection_results.rects.pop(index))
        detection_results.centers.pop(index)
        ## at the end, we add another center.
        fruit.centers.append(min)
        update_falling(fruit)


def box2window(box):
    return (box[0][0], box[0][1],
     box[1][0] - box[0][0],
     box[1][1] - box[0][1])
