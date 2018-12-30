import cv2
import numpy as np
from Fruit import Fruit
import SavedVideoRun as ni

NUM_OF_FRAMES = 5
MOVEMENT_RADIUS = 300
RESIZE_WINDOW_FACTOR = 0.2

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
    '''
    updates the is_falling parameter of fruit if it is falling.
    '''
    assert len(fruit.centers) > 1
    if fruit.centers[0][1] < fruit.centers[1][1]:
        fruit.is_falling = True

def track_object(detection_results, fruit):
    '''
    responsible for tracking the fruits. it suppose to be smart - which means to set some thresholds for
    wether we found the fruit.
    :param detection_results: the data from DETECTION
    :param fruit: the fruit object we know from previous frames.
    :return: true if found the fruit in the next frame, false otherwise.
    '''
    if len(detection_results.centers) > 0:
        x, y, w, h = fruit.track_window
        r = [(x, y), (x+w, y+h)]
        r_cent = center(r)
        n = len(detection_results.centers)
        min = detection_results.centers[0]
        min_dis = dis(r_cent, min)
        index = 0
        # finds the contour with minimal distance to tracker results.
        for i in range(1,n):
            if dis(detection_results.centers[i], r_cent) < min_dis:
                min = detection_results.centers[i]
                min_dis = dis(r_cent, min)
                index = i
        # threshold - if the fruit found is too far from original fruit.
        if min_dis > MOVEMENT_RADIUS:
            print("min dis: " + str(min_dis))
            return False
        else:
            detection_results.conts.pop(index)  ##removes the contour so we won't look for it again.
            old_track_window = box2window(detection_results.rects.pop(index))
            new_track_window = resize_track_window(old_track_window)
            fruit.track_window = new_track_window
            detection_results.centers.pop(index)
            ## at the end, we add another center.
            fruit.centers.append(min)
            update_falling(fruit)
            return True

def resize_track_window(track_window):
    '''
    for tracking with inner histogram.
    '''
    x, y, w, h = track_window
    factor = RESIZE_WINDOW_FACTOR
    inner_window = (int(x + factor*w), int(factor*h + y), int((1-2*factor)*w), int((1-2*factor)*h))
    return inner_window

def box2window(box):
    return (box[0][0], box[0][1],
     box[1][0] - box[0][0],
     box[1][1] - box[0][1])
