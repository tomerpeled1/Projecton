########
# first attempt - track multiple objects using mouse -
#  based on code from kalman filter.
########

#####
import cv2
import argparse
import time
import sys
import math
import numpy as np
import FruitDetection as fd

#####
NUM_OF_FRAMES = 50
WINDOW = 0
HIST = 1
COUNTER = 2
WINDOW_NAME = "MultiTracking"
keep_processing = True
selection_in_progress = False  # for mouse

##### parse args
parser = argparse.ArgumentParser(description='Perform ' + sys.argv[
    0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int,
                    help="specify camera to use", default=0)
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?',
                    help='specify optional video file')
args = parser.parse_args()

##### global variables
boxes = []
box = []

##### mouse selction
# current_mouse_position = np.ones(2, dtype=np.int32)
#
#
# def on_mouse(event, x, y, flags, params):
#     global boxes
#     global selection_in_progress
#     global box
#     current_mouse_position[0] = x
#     current_mouse_position[1] = y
#
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         box = []
#         sbox = [x, y]
#         selection_in_progress = True
#         box.append(sbox)
#
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         ebox = [x, y]
#         selection_in_progress = False
#         box.append(ebox)
#         boxes.append(box)
#

##### find center of rectangle
def center(points):
    # return a x and y position
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0
    return np.array([np.float32(x), np.float32(y)], np.float32)


def nothing(x):
    pass


##### define video capture object
cap = cv2.VideoCapture()


def track_objects(boxes, frame, cropped):
    global keep_processing
    start_t = cv2.getTickCount()
    # if cap.isOpened():
    #     ret, frame = cap.read()

    ### it didn't work to read from taskbar so i put it manually.
    s_lower = 60
    s_upper = 255
    v_lower = 32
    v_upper = 255
    ### get the hue histogram from every new fruit.
    for i in range(len(boxes)):
        if ((boxes[0][0][1] < boxes[0][1][1]) and (
            boxes[0][0][0] < boxes[0][1][0])):
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
                    {"window": track_window, "hist": crop_hist, "counter": 0})

            ### finished dealing with box, now free it.
            boxes.remove(boxes[0])
    ### in this section we calculate the new place of every fruit every frame.
    if cropped:
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for d in data:
            img_bproject = cv2.calcBackProject([img_hsv], [0, 1], d["hist"],
                                               [0, 180, 0, 255], 1)
            ret, track_window = cv2.meanShift(img_bproject, d["window"],
                                              term_crit)  ##credit for eisner
            d["window"] = track_window
            d["counter"] += 1
            if d["counter"] >= NUM_OF_FRAMES:
                data.remove(d)

        for i in range(len(data)):
            x, y, w, h = data[i]["window"]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (255, 0, 0), 2)

    else:  ##havent cropped yet
        pass

    ### after processing, show the image.
    cv2.imshow(WINDOW_NAME, frame)
    ## shit for timing
    stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency())
    key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF
    if (key == ord('x')):
        keep_processing = False
    elif (key == ord('f')):
        cv2.setWindowProperty("MultiTracking", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    # if frame_number > 130: ## just to pass the start of the video
    cv2.waitKey(0)

    return cropped


#### start the whole shit. if checks if there is a video or camera, else exit the program.
if args.video_file and cap.open(str(args.video_file)):
    # or (cap.open(args.camera_to_use))):

    ################################
    # initialize arrays for windows and histrograms.
    data = [] # holds lists. list[0] is the window, list[1] is the histogram, list[2] is a counter.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) #the window to show
    ################################

    cropped = False

    ## initialize shit for meanshift, maybe play with it.
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    ##

    if cap.isOpened():
        ret, bg = cap.read()

    ret, frame = cap.read()
    conts, rects = fd.fruit_detection(frame, bg, 4100)

    #### this is the main part which runs the program:
    while keep_processing:
        if cap.isOpened():
            ret, frame = cap.read()
        cropped = track_objects(rects,frame,cropped)

    ##### after the proccecing, close all the windows
    cv2.destroyAllWindows()

else:
    print("Camera is not connected and Video doesn't work")
