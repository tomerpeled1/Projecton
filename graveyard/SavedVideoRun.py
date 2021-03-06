from graveyard import VideoInterface as vi
import FruitDetection as fd
import RealTimeTracker as rtt
import cv2
import time
import numpy as np
from Fruit import Fruit

## parameters for meanshift
s_lower = 60
s_upper = 255
v_lower = 32
v_upper = 255

##consts
MINIMUM_NUM_OF_CENTERS_TO_EXTRACT = 4
CONTOUR_AREA_THRESH = 1000
NUM_OF_FRAMES = 5
MAX_NUM_OF_FRAMES_ON_SCREEN = 13
WINDOW_NAME = "Fruit tracker"
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
##init window
# cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # the window to show

# HISTS_THRESH = 0.4  # the old one
HISTS_THRESH = 0.1
HISTS_COMPARE_METHOD = cv2.HISTCMP_CORREL


def center(box):
    '''
    returns center of a box.
    '''
    # return a x and y position
    x = box[0][0] + box[0][1] / 2.0
    y = box[1][0] + box[1][1] / 2.0
    return np.array([np.float32(x), np.float32(y)], np.float32)


def draw_rectangles(fruit, frame, color, size=2):
    '''
    draws the current tracking windows on the frame
    '''
    x, y, w, h = fruit.track_window
    return cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=size)


def draw_center(fruit, frame):
    '''
    Draws the center of a single fruit.
    :param fruit: fruit to draw it's center.
    :param frame: the frame in which we want to draw.
    '''
    for center in fruit.centers:
        cv2.circle(frame, center, 2, (0, 0, 255), -1)


def calculate_hist_window(window, img_hsv):
    '''
    calculates the histogram of an image in hsv.
    :param cropped: the image which histogram we want to calculate.
    '''
    x, y, w, h = window
    cropped = img_hsv[y:y + h, x:x + w].copy()
    # cv2.imshow("histogram", cropped)
    mask = cv2.inRange(cropped, np.array((0., float(s_lower), float(v_lower))),
                       np.array((180., float(s_upper), float(v_upper))))  # TODO understand parameters.
    crop_hist = cv2.calcHist([cropped], [0, 1], mask, [180, 255],
                             [0, 180, 0, 255])
    cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)
    return crop_hist


def calc_meanshift_all_fruits(fruits_info, img_hsv):
    # does the proccess of calculating the new meanshift every frame.
    # compares the hist to the known one to see if the fruit has left the screen.
    fruits_to_extract = []
    for fruit in fruits_info:
        x, y, w, h = fruit.track_window
        if len(fruit.centers) > 1:
            # cut the image - search only above the fruit if it is rising and below it if it is falling.
            if not fruit.is_falling:
                img_bproject = cv2.calcBackProject([img_hsv[:y + h, :]], [0, 1], fruit.hist, [0, 180, 0, 255], 1)
            else:
                img_bproject = cv2.calcBackProject([img_hsv[y:, :]], [0, 1], fruit.hist, [0, 180, 0, 255], 1)
        else:
            img_bproject = cv2.calcBackProject([img_hsv], [0, 1], fruit.hist, [0, 180, 0, 255], 1)

        ret, track_window = cv2.meanShift(img_bproject, fruit.track_window, term_crit)  ##credit for eisner
        new_hist = calculate_hist_window(track_window, img_hsv)
        dis = cv2.compareHist(new_hist, fruit.hist, HISTS_COMPARE_METHOD)
        if (abs(dis) > HISTS_THRESH) and fruit.counter < MAX_NUM_OF_FRAMES_ON_SCREEN:  # threshold for hist resemblance.
            fruit.track_window = track_window
            fruit.counter += 1
        else:
            fruits_info.remove(fruit)
            if not fruit.is_falling and len(fruit.centers) > MINIMUM_NUM_OF_CENTERS_TO_EXTRACT:
                fruits_to_extract.append(fruit)
    print_and_extract_centers(fruits_to_extract)


def print_and_extract_centers(fruits_to_extract):
    '''
    :param fruits_to_extract: a list of friuts that we want to move to algorithms - to calculate their routes.
    the connection to algorithms module.
    '''
    if fruits_to_extract:
        # sc.create_slice(fruits_to_extract)
        print("centers of:" + str([fruit.centers for fruit in fruits_to_extract]))


def get_hists(detection_results, frame):
    '''
    returns the data known about the detected fruits.
    :param detection_results: the results of the detection.
    :param frame: the frame in which we want to find out data.
    :return: the information known about the detected fruits (list of fruits).
    '''
    fruits_info = []
    boxes = detection_results.rects
    while len(boxes) > 0:
        # crop = frame[boxes[0][0][1]:boxes[0][1][1], boxes[0][0][0]:boxes[0][1][0]].copy() #crop the box.
        # h, w, c = crop.shape
        # if (h > 0) and (w > 0): #if the box isn't empty.
        cropped = True
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        track_window = (boxes[0][0][0], boxes[0][0][1],
                        boxes[0][1][0] - boxes[0][0][0],
                        boxes[0][1][1] - boxes[0][0][1])
        crop_hist = calculate_hist_window(track_window, hsv_frame)
        # after calculating the histrogram of the fruit, we add it to the big array and the window to the big array.
        fruits_info.append(Fruit(track_window, crop_hist, 0, [detection_results.centers[0]], detection_results.time_created))
        # finished dealing with box, now free it.
        detection_results.pop_element(0)
    return fruits_info


def background_and_wait(cap):
    '''
    returns a frame of the background without fruits. change the numbers to use other videos.
    :param cap: the stream of the video.
    :return: the background.
    '''
    vi.wait(0.0, cap)
    bg = vi.get_background(cap)
    vi.wait(0, cap)
    return bg


def track_known_fruits(fruits_info, current_frame, detection_results):
    '''
    tracks all fruits which we already found previously.
    :param fruits_info: the list of known fruits.
    :param current_frame: the frame in which we want to update the tracker.
    :param detection_results: the fruits detected in the current frame.
    '''
    img_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)  # turn image to hsv.
    calc_meanshift_all_fruits(fruits_info, img_hsv)  # calculate the meanshift for all fruits.
    for fruit in fruits_info:
        current_frame = draw_rectangles(fruit, current_frame, (255, 192, 203), 5)
    if (len(detection_results.conts) > 0):
        toDelete = []
        for fruit in fruits_info:
            if rtt.track_object(detection_results, fruit): #if we decided that we found the right contour
                if (fruit.counter <= 3): #update the histogram for tracker, when fruit enters the screen
                                        #  (when the fruit enters we find a part of it and it is harder to track).
                    # cv2.imshow("hsv new", img_hsv)
                    fruit.hist = calculate_hist_window(fruit.track_window, img_hsv)
            else: # didnt find the fruit so delete it
                toDelete.append(fruit)
        fruits_to_extract = []
        for deleted_fruit in toDelete: # decide if deleted fruit is good to move to algorithms.
            if len(deleted_fruit.centers) > MINIMUM_NUM_OF_CENTERS_TO_EXTRACT and not deleted_fruit.is_falling:
                fruits_to_extract.append(deleted_fruit)
            fruits_info.remove(deleted_fruit)
        print_and_extract_centers(fruits_to_extract)


def insert_new_fruits(detection_results, fruits_info, current):
    '''
    detection of new fruits which entered screen.
    :param detection_results: detected new fruits.
    :param fruits_info: list of fruits known.
    :param current: the frame in which we are looking.
    '''
    fruits_info += get_hists(detection_results, current)


def run_detection(video_name):
    fruits_info = []
    cap = cv2.VideoCapture(video_name)
    bg = background_and_wait(cap)
    current = 0
    cv2.imshow("aaaa", bg)
    times = []
    while cap.isOpened():
        t2 = time.perf_counter()
        current = vi.get_background(cap)
        detection_results = fd.fruit_detection(current, bg, CONTOUR_AREA_THRESH)
        cv2.drawContours(current, detection_results.conts, -1, (0, 255, 0), 2)
        track_known_fruits(fruits_info, current,
                           detection_results)  # calculates meanshift for fruits known. removes fruits which left frame.
        if len(detection_results.conts) > 0:
            insert_new_fruits(detection_results, fruits_info, current)
        for fruit in fruits_info:
            if not fruit.is_falling:
                draw(fruit, current)
        # print("len of fruits: " + str(len(fruits_info)))
        t1 = time.perf_counter()
        times.append(abs(t2 - t1))
        cv2.imshow("frame", current)
        if len(times) == 200:
            break
        # print("time for thing we check now: " + str(abs(t2 - t1)))
        cv2.waitKey(0)
    # print("avg : " + str(sum(times)/len(times)))
    # print("max : " + str(max(times)))


def draw(fruit, frame):

    '''
    auto generated function, does some things and displays shit.
    '''
    draw_rectangles(fruit, frame, (255, 0, 0))
    draw_center(fruit, frame)  # conts_and_rects holds all the centers of all fruits - it has a list of lists.


if __name__ == '__main__':
    run_detection("SmallFruit2.flv")
