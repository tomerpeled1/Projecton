import VideoInterface as vi
import FruitDetection as fd
import RealTimeTracker as rtt
import cv2
import argparse
import time
import sys
import math
import numpy as np

## parameters for meanshift
s_lower = 60
s_upper = 255
v_lower = 32
v_upper = 255

##consts
NUM_OF_FRAMES = 5
WINDOW_NAME = "Fruit tracker"
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
##init window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # the window to show


def center(box):
    '''
    returns center of a box.
    '''
    # return a x and y position
    x = box[0][0] + box[0][1] / 2.0
    y = box[1][0] + box[1][1] / 2.0
    return np.array([np.float32(x), np.float32(y)], np.float32)


def draw_rectangles(data, frame):
    '''
    draws the current tracking windows on the frame
    '''
    for i in range(len(data)):
        x, y, w, h = data[i]["window"]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (255, 0, 0), 2)
    return frame

def draw_centers(data, frame):
    for d in data:
        for center in d["centers"]:
            cv2.circle(frame, center, 2, (0, 0, 255), -1)


def calc_meanshift_all_fruits(data, img_hsv):
    # does the proccess of calculating the new meanshift every frame.
    for d in data:
        img_bproject = cv2.calcBackProject([img_hsv], [0, 1], d["hist"],
                                           [0, 180, 0, 255], 1)
        ret, track_window = cv2.meanShift(img_bproject, d["window"],
                                          term_crit)  ##credit for eisner
        d["window"] = track_window
        d["counter"] += 1
        if d["counter"] >= NUM_OF_FRAMES:
            print_centers(d["centers"])
            data.remove(d)
        x, y, w, h = track_window


def print_centers(arr):
    print("centers of:" + str(arr))


def get_hists(boxes, data, frame):
    # gets histrograms and initial tracking windows from a list of boxes and a certain frame.
    crop = frame[boxes[0][0][1]:boxes[0][1][1],
           boxes[0][0][0]:boxes[0][1][0]].copy()
    h, w, c = crop.shape
    if (h > 0) and (w > 0):
        cropped = True
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_crop, np.array(
            (0., float(s_lower), float(v_lower))), np.array(
            (180., float(s_upper), float(v_upper))))

        crop_hist = cv2.calcHist([hsv_crop], [0, 1], mask, [180, 255],
                                 [0, 180, 0, 255])
        cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)
        track_window = (boxes[0][0][0], boxes[0][0][1],
                        boxes[0][1][0] - boxes[0][0][0],
                        boxes[0][1][1] - boxes[0][0][1])

        ##after calculating the histrogram of the fruit, we add it to the big array and the window to the big array.
        data.append(
            {"window": track_window, "hist": crop_hist, "counter": 0, "centers": []})

    ### finished dealing with box, now free it.
    boxes.remove(boxes[0])


def run_detection(video_name):
    # runs the shit. the main() of the program.
    data = []
    cap = cv2.VideoCapture(video_name)
    vi.wait(6, cap)
    background = vi.get_background(cap)
    vi.wait(4, cap)
    frame = vi.get_background(cap)
    conts_and_rects = fd.fruit_detection(frame, background, 3000) #runs detection on the first frame.
    get_hists(conts_and_rects.rects, data, frame)
    null = draw_rectangles(data, frame)
    cv2.waitKey(0)
    counter = 0
    #### main loop
    while counter < 100 * NUM_OF_FRAMES: # proccess that runs on every frame, updates the prediction and
        if cap.isOpened():
            frame = vi.get_background(cap)
        # meanShift
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        calc_meanshift_all_fruits(data, img_hsv)
        #detection
        conts_and_rects = fd.fruit_detection(frame, background, 4000)
        cv2.drawContours(frame, conts_and_rects.conts, -1, (0, 255, 0), 2)

        #track - get centers from fruits and match it to the right fruit.
        if len(conts_and_rects.centers) > 0:
            rtt.stupid_tracker(conts_and_rects, data)
        # now that stupid_tracker left in conts_and_rects only new fruits, we add them to data and to meanshift search cycle.
        if len(conts_and_rects.centers) != 0:
            get_hists(conts_and_rects.rects, data, frame)


        counter, frame = run_shit_for_display(counter, data, frame)


def run_shit_for_display(counter, data, frame):
    '''
    auto generated function, does some things and displays shit.
    '''
    frame = draw_rectangles(data, frame)
    draw_centers(data, frame) # conts_and_rects holds all the centers of all fruits - it has a list of lists.
    cv2.imshow(WINDOW_NAME, frame)
    counter += 1
    cv2.waitKey(0)
    return counter, frame




if __name__ == '__main__':
    run_detection("first_video.mp4")
