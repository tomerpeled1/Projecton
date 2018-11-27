import VideoInterface as vi
import FruitDetection as fd
import cv2
import argparse
import time
import sys
import math
import numpy as np

s_lower = 60
s_upper = 255
v_lower = 32
v_upper = 255

NUM_OF_FRAMES = 5

WINDOW_NAME = "window1"

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # the window to show

def center(box):
    # return a x and y position
    x = box[0][0] + box[0][1] / 2.0
    y = box[1][0] + box[1][1] / 2.0
    return np.array([np.float32(x), np.float32(y)], np.float32)

def draw_rectangles(data, frame):
    for i in range(len(data)):
        x, y, w, h = data[i]["window"]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (255, 0, 0), 2)
    return frame


def calc_meanshift_all_fruits(data, img_hsv):
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
        d["centers"].append((x + h/2.0, y + w/2.0))

def print_centers(arr):
    print("centers of:" + str(arr))

def get_hists(boxes, data, frame):
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
    data = []
    cap = cv2.VideoCapture(video_name)
    vi.wait(6, cap)
    ret, background = cap.retrieve()
    vi.wait(17,cap)
    ret, frame = cap.retrieve()
    conts, boxes = fd.fruit_detection(frame, background, 3000)

    cv2.drawContours(frame, conts, -1, (0, 255, 0), 2)
    for i in range(len(boxes)):
        frame = cv2.rectangle(frame, boxes[i][0], boxes[i][1],
                              (255, 0, 0), 2)
    cv2.waitKey(0)
    for i in range(len(boxes)):
        get_hists(boxes, data, frame)

    null = draw_rectangles(data, frame)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(0)

    counter = 0
    while counter < 10*NUM_OF_FRAMES:
        start_t = cv2.getTickCount()
        if cap.isOpened():
            ret, frame = cap.read()
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        calc_meanshift_all_fruits(data, img_hsv)

        frame = draw_rectangles(data, frame)

        cv2.imshow(WINDOW_NAME, frame)
        counter += 1
        cv2.waitKey(0)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency())
        print(stop_t)




run_detection("first_video.mp4")