"""
the main image processing file - does the tracking and gets detection from FruitDetection.
works in pixels coordinates - (0,0) is top left of the frame.
"""


# import FruitDetection as Fd
import RealTimeTracker as Rtt
# from CameraInterface import Camera
# import CameraInterface as Ci
import Algorithmics as Algo
import cv2
# import time
import numpy as np
from Fruit import Fruit


forcheck = 0

MINIMUM_NUM_OF_CENTERS_TO_EXTRACT = 4  # Minimal number of centers for fruit to create good fit.

# Parameters for meanshift
s_lower = 60
s_upper = 255
v_lower = 32
v_upper = 255
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)

# Consts
SAVED_VIDEO_NAME = "sundayNoon.flv"
AD_IMAGE = "AD_TEMPLATE.png"
CONTOUR_AREA_THRESH = 1000  # Threshold for fruit detection.
MAX_NUM_OF_FRAMES_ON_SCREEN = 13  # Maximal number of frames for fruit to remain on screen.
FRUIT_TO_EXTRACT = []  # List of fruits needs to be extracted.

HISTS_THRESH = 0.2  # Threshold for difference between histograms for recognition of fruits.
HISTS_COMPARE_METHOD = cv2.HISTCMP_CORREL  # Method to compare between histograms.

INTEGRATE_WITH_ALGORITHMICS = False

# Fruits to print on screen for debugging of the algorithms.
fruits_for_debug_trajectories = []


def draw_rectangle(fruit, frame, color, size=2):
    """
    Draws the track window of the fruit in the frame.
    :param fruit: the fruit in the frame
    :param frame: the frame to draw on
    :param color: color of rectangle
    :param size: thickness of rectangle
    :return:
    """
    x, y, w, h = fruit.track_window
    return cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=size)


def draw_center(fruit, frame):
    """
    Draws the centers of a single fruit.
    :param fruit: fruit to draw its centers.
    :param frame: the frame in which we want to draw.
    """
    for cen in fruit.centers:
        cv2.circle(frame, cen[0][0:2], 2, (0, 0, 255), -1)


def draw_trajectory(fruit, frame):
    """
    draws the predicted route of the fruit  and the centers we found for it
    on the specific frame. used mostly for debugging.
    :param fruit: a fruit object
    :param frame: a single frame
    """
    # ---------get the centers and transfer to cm.-------#
    centers_cm = [Algo.pixel2cm(center[0]) for center in fruit.centers]
    x_coords = [center[0] for center in centers_cm]
    y_coords = [center[1] for center in centers_cm]
    # t_coords = [center[2] for center in centers_cm]
    times_centers = range(len(x_coords))

    # ------- get the trajectory of the fruit -------#
    t_tot = 3
    dt = 0.02
    times_trajectory = range(-int(t_tot / dt), int(t_tot / dt))
    xy_cm = [[0 for _ in times_trajectory], [0 for _ in times_trajectory]]
    xy_pixels = [[0 for _ in times_trajectory], [0 for _ in times_trajectory]]
    route = fruit.trajectory.calc_trajectory()

    # -------draw fitted trajectory----------#
    for i in times_trajectory:
        if i==90:
            k=0
        xy_cm[0][i], xy_cm[1][i] = route(dt * i)
        xy_pixels[1][i], xy_pixels[0][i], t = Algo.cm2pixel((xy_cm[0][i], xy_cm[1][i], dt * i))
        xy_pixels[1][i] = Algo.FRAME_SIZE[0] - xy_pixels[1][i]
        xy_pixels[0][i] = Algo.FRAME_SIZE[1] - xy_pixels[0][i]
        cv2.circle(frame, (int(xy_pixels[0][i]), int(xy_pixels[1][i])), 2, (255, 0, 255), -1)

    # ---------draw the centers of the fruits------------#
    xy_centers = [[0 for _ in times_centers], [0 for _ in times_centers]]
    cens_original = []
    for cen in fruit.centers:
        cens_original.append(cen[:-1])
    for i in times_centers:
        xy_centers[1][i], xy_centers[0][i], t = Algo.cm2pixel((x_coords[i], y_coords[i], dt * i))
        xy_centers[1][i] = Algo.FRAME_SIZE[0] - xy_centers[1][i]
        xy_centers[0][i] = Algo.FRAME_SIZE[1] - xy_centers[0][i]
        cv2.circle(frame, (int(xy_centers[0][i]), int(xy_centers[1][i])), 3, (0, 255, 255), -1)
        if (i==times_centers[-1]):
            K=0



def calculate_hist_window(window, img_hsv):
    """
    Calculates the histogram of a window in image. The histogram is calculated by the hue.
    :param window: area to calculate histogram, given in (x,y,w,h)
    :param img_hsv: image of frame in hsv
    :return: the histogram of the window.
    """
    x, y, w, h = window
    cropped = img_hsv[y:y + h, x:x + w].copy()
    mask = cv2.inRange(cropped, np.array((0., float(s_lower), float(v_lower))),
                       np.array((180., float(s_upper), float(v_upper))))
    crop_hist = cv2.calcHist([cropped], [0], mask, [180],
                             [0, 180])
    cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)
    return crop_hist


def calc_meanshift_all_fruits(fruits_info, img_hsv):
    """
    Calculates the meanshift for all fruits on screen.
    Calculates the difference between the histograms of the fruits between the frames and makes sure it passes
    HISTS_THRESH. If it does it updates the track window for fruit. Otherwise we had lost the fruit and add it to
    FRUIT_TO_EXTRACT.
    :param fruits_info: list of fruits info
    :param img_hsv: The frame in hsv form
    """
    global FRUIT_TO_EXTRACT
    for fruit in fruits_info:
        x, y, w, h = fruit.track_window
        if len(fruit.centers) > 1:
            # Checks if fruit is falling or not to know in what half of the frame to look for it.
            if not fruit.is_falling:
                img_bproject = cv2.calcBackProject([img_hsv[:y + h, :]], [0, 1], fruit.hist, [0, 180, 0, 255], 1)
            else:
                img_bproject = cv2.calcBackProject([img_hsv[y:, :]], [0, 1], fruit.hist, [0, 180, 0, 255], 1)
        else:
            img_bproject = cv2.calcBackProject([img_hsv], [0, 1], fruit.hist, [0, 180, 0, 255], 1)
        # Calculation of the new track window by meanshift algorithm.
        ret, track_window = cv2.meanShift(img_bproject, fruit.track_window, term_crit)  # credit for eisner
        # Calculates the new histogram of the fruit.
        new_hist = calculate_hist_window(track_window, img_hsv)
        # Calculated correlation between new histogram to previous one.
        correlation = cv2.compareHist(new_hist, fruit.hist, HISTS_COMPARE_METHOD)
        # If the correlation is high enough we update the track window.
        if (abs(correlation) > HISTS_THRESH) and fruit.counter < MAX_NUM_OF_FRAMES_ON_SCREEN:  # threshold for histogram
            # resemblance
            fruit.track_window = track_window
            fruit.hist = new_hist
            fruit.correlation = correlation
            # fruit.centers[fruit.counter] = fruit.centers[fruit.counter][:-1] + (correlation,)
            fruit.counter += 1
        else:  # Otherwise the fruit is gone and we remove it from fruits_info and add it to FRUIT_TO_EXTRACT
            fruits_info.remove(fruit)
            if not fruit.is_falling and len(fruit.centers) > MINIMUM_NUM_OF_CENTERS_TO_EXTRACT:
                FRUIT_TO_EXTRACT.append(fruit)
    update_trajectories(FRUIT_TO_EXTRACT)

def clear_fruits():
    global FRUIT_TO_EXTRACT
    FRUIT_TO_EXTRACT[:] = []

def update_trajectories(fruits_to_extract):
    """
    Extracts the centers list of each fruit to algorithm module (without first and last because they are unreliable).
    Prints the centers for debugging purposes.
    :param fruits_to_extract: list of fruits to be extracted.
    """
    for fruit in fruits_to_extract:
        fruit.centers = fruit.centers[1:-1]
        fruit.counter -= 2 ## TODO maybe huge bug!
    if fruits_to_extract:
        # # ---------Add trajectory to fruit object ------- #
        global fruits_for_debug_trajectories
        fruits_and_trajectories = []
        for fruit in fruits_to_extract:
            centers_cm = [Algo.pixel2cm(center[0]) for center in fruit.centers]
            fruit.trajectory = Algo.get_trajectory_by_fruit_locations(centers_cm)
            # --- add first fruit to debug fruits buffer ---#
            fruits_for_debug_trajectories.append(fruit)
            # fruits_and_trajectories.append(fruit)
        # if INTEGRATE_WITH_ALGORITHMICS:
        #     Algo.add_slice_to_queue(fruits_to_extract)
        # global FRUIT_TO_EXTRACT
        # FRUIT_TO_EXTRACT[:] = []


def get_fruits_info(detection_results, frame):
    """
    Returns the data known about the detected unknown fruits (new fruits only!).
    :param detection_results: the results of the detection. DetectionResults object.
    :param frame: the frame in which we want to find out data.
    :return: the information known about the detected fruits (list of fruits).
    """
    fruits_info = []
    boxes = detection_results.rects
    while len(boxes) > 0:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Turns the first box to track window.
        track_window = (boxes[0][0][0], boxes[0][0][1],
                        boxes[0][1][0] - boxes[0][0][0],
                        boxes[0][1][1] - boxes[0][0][1])
        # Resize the track window so it will be bounded by fruit.
        track_window = Rtt.resize_track_window(track_window)
        # Calculate the histogram of the window.
        crop_hist = calculate_hist_window(track_window, hsv_frame)
        # After calculating the histogram of the fruit, we add it to the list of fruits.
        fruits_info.append(Fruit(track_window, crop_hist, 0,
                                 [(detection_results.centers[0], 1)], detection_results.time_created))
        # Finished dealing with box, now free it.
        detection_results.pop_element(0)
    return fruits_info


def track_known_fruits(fruits_info, current_frame, detection_results):
    """
    Tracks all fruits which we already found previously.
    :param fruits_info: The list of known fruits.
    :param current_frame: The frame in which we want to update the tracker.
    :param detection_results: The fruits detected in the current frame.
    """
    img_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)  # turn image to hsv.
    calc_meanshift_all_fruits(fruits_info, img_hsv)  # calculate meanshift of all fruits and update their track windows
    # draw new track windows of all fruits.
    for fruit in fruits_info:
        current_frame = draw_rectangle(fruit, current_frame, (255, 20, 147), 5)

    # track all known fruits - match their new track window to the contours in detection results.
    if len(detection_results.conts) > 0:
        to_delete = []
        for fruit in fruits_info:
            # Try to track fruit and if found update its histogram.
            if not Rtt.track_object(detection_results, fruit):
                to_delete.append(fruit) # update tracker using the detection results.
                # if fruit.counter <= MAX_NUM_OF_FRAMES_ON_SCREEN:
                    # fruit.hist = calculate_hist_window(fruit.track_window, img_hsv) ## TODO remove first fruits for trajectory fit
            # If fruit not found extract it.
            # else:
            #     to_delete.append(fruit)
        # Extract all fruits not tracked.
        global FRUIT_TO_EXTRACT
        for deleted_fruit in to_delete:
            if len(deleted_fruit.centers) > MINIMUM_NUM_OF_CENTERS_TO_EXTRACT and not deleted_fruit.is_falling:
                FRUIT_TO_EXTRACT.append(deleted_fruit)
            fruits_info.remove(deleted_fruit)
        update_trajectories(FRUIT_TO_EXTRACT)


def insert_new_fruits(detection_results, fruits_info, current):
    """
    Detection of new fruits which entered screen.
    :param detection_results: Detected new fruits.
    :param fruits_info: List of fruits known.
    :param current: The frame in which we are looking.
    """
    fruits_info += get_fruits_info(detection_results, current)


def debug_with_buffer(buffer):
    """
    Debugging method which shows images and allows you to pass easily between them.
    :param buffer: List of frames.
    """
    i = 0
    while True:
        for fruit in fruits_for_debug_trajectories:
            # draw_center(fruit, buffer[i])
            draw_trajectory(fruit, buffer[i])
        cv2.imshow("debug", buffer[i])
        x = cv2.waitKey(1)
        if x == 49:  # '1' key
            i -= 1
        elif x == 50:  # '2' key
            i += 1


def show_original(camera):
    """
    Show original frames taken by camera.
    :param camera: Camera object of camera
    """
    debug_with_buffer(camera.buffer)


def draw(fruit, frame):
    """
    Draws center and track window for fruit.
    :param fruit: Fruit object
    :param frame: frame where fruit appears
    """
    draw_rectangle(fruit, frame, (255, 0, 0))
    draw_center(fruit, frame)


def check_ad(frame):
    template = cv2.imread(AD_IMAGE)
    w, h = template.shape[1], template.shape[0]
    method = eval('cv2.TM_CCOEFF_NORMED')
    # Apply template Matching

    cv2.imshow("frame", frame)
    cv2.imshow("template", template)

    res = cv2.matchTemplate(frame, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)

    print(max_val)

    # cv2.rectangle(frame, top_left, bottom_right, 255, 2)
    # plt.subplot(121), plt.imshow(res, cmap='gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(frame, cmap='gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.show()

    if max_val > 0.9:
        print("there is an ad")
        return True
    print("no addddddddddddddddddddd")
    return False


def init_everything(integrate_with_algorithmics=INTEGRATE_WITH_ALGORITHMICS):
    """
    :param integrate_with_algorithmics: boolean that decides weather to integrate with algorithmics or not.
    """
    global INTEGRATE_WITH_ALGORITHMICS
    INTEGRATE_WITH_ALGORITHMICS = integrate_with_algorithmics
