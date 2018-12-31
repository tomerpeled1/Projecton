import FruitDetection as Fd
import RealTimeTracker as Rtt
from CameraInterface import Camera
import CameraInterface as Ci
import SliceCreator as Sc
import cv2
import time
import numpy as np
from Fruit import Fruit

# parameters for meanshift
MINIMUM_NUM_OF_CENTERS_TO_EXTRACT = 4
s_lower = 60
s_upper = 255
v_lower = 32
v_upper = 255

# consts
SAVED_VIDEO_NAME = "SmallFruit2.flv"
CONTOUR_AREA_THRESH = 1000
NUM_OF_FRAMES = 5
MAX_NUM_OF_FRAMES_ON_SCREEN = 13
WINDOW_NAME = "Fruit tracker"
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)
FRUIT_TO_EXTRACT = []
# init window
# cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # the window to show22

HISTS_THRESH = 0.2
HISTS_COMPARE_METHOD = cv2.HISTCMP_CORREL

# Magic numbers for camera
SECONDS_FOR_BG = 3

INTEGRATE_WITH_ALGORITHMICS = True

fruits_for_debug_trajectories = []


def center(box):
    """
    returns center of a box.
    """
    # return a x and y position
    x = box[0][0] + box[0][1] / 2.0
    y = box[1][0] + box[1][1] / 2.0
    return np.array([np.float32(x), np.float32(y)], np.float32)


def draw_rectangles(fruit, frame, color, size=2):
    """
    draws the current tracking windows on the frame
    """
    x, y, w, h = fruit.track_window
    return cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=size)


def draw_center(fruit, frame):
    """
    Draws the center of a single fruit.
    :param fruit: fruit to draw it's center.
    :param frame: the frame in which we want to draw.
    """
    for cen in fruit.centers:
        cv2.circle(frame, cen[:-1], 2, (0, 0, 255), -1)


def draw_trajectory(fruit, frame):
    centers_cm = [Sc.pixel2cm(center) for center in fruit.centers]
    x_coords = [fruit_loc[0] for fruit_loc in centers_cm]  # TODO this is a bug. need to make in a loop
    y_coords = [fruit_loc[1] for fruit_loc in centers_cm]
    t_coords = [fruit_loc[2] for fruit_loc in centers_cm]
    times_centers = range(len(x_coords))

    # if fruit.trajectory:
    #     cv2.putText(frame, 'SHANINJA', (200, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    T = 3
    dt = 0.02
    times_trajectory = range(-int(T / dt), int(T / dt))
    xy_cm = [[0 for _ in times_trajectory], [0 for _ in times_trajectory]]
    xy_pixels = [[0 for _ in times_trajectory], [0 for _ in times_trajectory]]
    route = fruit.trajectory.calc_trajectory()
    # draw fitted trajectory
    for i in times_trajectory:
        xy_cm[0][i], xy_cm[1][i] = route(dt * i)
        xy_pixels[1][i], xy_pixels[0][i], t = Sc.cm2pixel((xy_cm[0][i], xy_cm[1][i], dt * i))
        cv2.circle(frame, (int(xy_pixels[0][i]), int(xy_pixels[1][i])), 2, fruit.color, -1)

    # draw the centers of the fruits
    xy_centers = [[0 for _ in times_centers], [0 for _ in times_centers]]
    for i in times_centers:
        xy_centers[1][i], xy_centers[0][i], t = Sc.cm2pixel((x_coords[i], y_coords[i], dt * i))
        cv2.circle(frame, (int(xy_centers[0][i]), int(xy_centers[1][i])), 5, (0, 0, 255), -1)


def calculate_hist_window(window, img_hsv):
    """
    calculates the histogram of an image in hsv.
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
    # does the process of calculating the new meanshift every frame.
    # compares the hist to the known one to see if the fruit has left the screen.
    global FRUIT_TO_EXTRACT
    for fruit in fruits_info:
        x, y, w, h = fruit.track_window
        if len(fruit.centers) > 1:
            if not fruit.is_falling:
                img_bproject = cv2.calcBackProject([img_hsv[:y + h, :]], [0, 1], fruit.hist, [0, 180, 0, 255], 1)
            else:
                img_bproject = cv2.calcBackProject([img_hsv[y:, :]], [0, 1], fruit.hist, [0, 180, 0, 255], 1)
        else:
            img_bproject = cv2.calcBackProject([img_hsv], [0, 1], fruit.hist, [0, 180, 0, 255], 1)

        # x, y, w, h = fruit.track_window
        # factor = 0.2
        # inner_window = (int(x + factor*w), int(factor*h + y), int((1-2*factor)*w), int((1-2*factor)*h))
        ret, track_window = cv2.meanShift(img_bproject, fruit.track_window, term_crit)  # credit for eisner
        new_hist = calculate_hist_window(track_window, img_hsv)
        dis = cv2.compareHist(new_hist, fruit.hist, HISTS_COMPARE_METHOD)
        # cv2.imshow("ggg", cropped_window)
        if (abs(dis) > HISTS_THRESH) and fruit.counter < MAX_NUM_OF_FRAMES_ON_SCREEN:  # threshold for hist resemblance.
            fruit.track_window = track_window
            fruit.counter += 1

        else:
            print("dis: " + str(dis))
            fruits_info.remove(fruit)
            if not fruit.is_falling and len(fruit.centers) > MINIMUM_NUM_OF_CENTERS_TO_EXTRACT:
                FRUIT_TO_EXTRACT.append(fruit)
    print_and_extract_centers(FRUIT_TO_EXTRACT)


def print_and_extract_centers(fruits_to_extract):
    for fruit in fruits_to_extract:
        fruit.centers = fruit.centers[1:-1]
    if fruits_to_extract:
        # ---------Add trajectory to fruit object ------- #
        global fruits_for_debug_trajectories
        for fruit in fruits_to_extract:
            centers_cm = [Sc.pixel2cm(center) for center in fruit.centers]
            fruit.trajectory = Sc.get_trajectory(centers_cm)
            # --- add fruit to debug fruits buffer ---#
            fruits_for_debug_trajectories.append(fruit)

        if (INTEGRATE_WITH_ALGORITHMICS):
            Sc.update_and_slice(fruits_to_extract)

        global FRUIT_TO_EXTRACT
        FRUIT_TO_EXTRACT[:] = []
        print("centers of:" + str([fruit.centers for fruit in fruits_to_extract]))


def get_hists(detection_results, frame):
    """
    returns the data known about the detected fruits.
    :param detection_results: the results of the detection.
    :param frame: the frame in which we want to find out data.
    :return: the information known about the detected fruits (list of fruits).
    """
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
        track_window = Rtt.resize_track_window(track_window)
        crop_hist = calculate_hist_window(track_window, hsv_frame)
        # after calculating the histrogram of the fruit, we add it to the big array and the window to the big array.
        fruits_info.append(
            Fruit(track_window, crop_hist, 0, [detection_results.centers[0]], detection_results.time_created))
        # finished dealing with box, now free it.
        detection_results.pop_element(0)
    return fruits_info


def track_known_fruits(fruits_info, current_frame, detection_results):
    """
    tracks all fruits which we already found previously.
    :param fruits_info: the list of known fruits.
    :param current_frame: the frame in which we want to update the tracker.
    :param detection_results: the fruits detected in the current frame.
    """
    img_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)  # turn image to hsv.
    calc_meanshift_all_fruits(fruits_info, img_hsv)  # calculate the meanshift for all fruits.
    for fruit in fruits_info:
        current_frame = draw_rectangles(fruit, current_frame, (255, 20, 147), 5)
    if len(detection_results.conts) > 0:
        to_delete = []
        for fruit in fruits_info:
            if Rtt.track_object(detection_results, fruit):  # update tracker using the detection results.
                if fruit.counter <= MAX_NUM_OF_FRAMES_ON_SCREEN:  # TODO check shit
                    # cv2.imshow("hsv new", img_hsv)
                    fruit.hist = calculate_hist_window(fruit.track_window, img_hsv)
            else:
                to_delete.append(fruit)
        global FRUIT_TO_EXTRACT
        for deleted_fruit in to_delete:
            if len(deleted_fruit.centers) > MINIMUM_NUM_OF_CENTERS_TO_EXTRACT and not deleted_fruit.is_falling:
                FRUIT_TO_EXTRACT.append(deleted_fruit)
            fruits_info.remove(deleted_fruit)
        print_and_extract_centers(FRUIT_TO_EXTRACT)


def insert_new_fruits(detection_results, fruits_info, current):
    """
    detection of new fruits which entered screen.
    :param detection_results: detected new fruits.
    :param fruits_info: list of fruits known.
    :param current: the frame in which we are looking.
    """
    fruits_info += get_hists(detection_results, current)


def run_detection(src, settings, live, crop, flip):
    # global Lock
    # Lock = False
    if INTEGRATE_WITH_ALGORITHMICS:
        Sc.init_everything()
    fruits_info = []
    camera = Camera(src, FLIP=flip, CROP=crop, LIVE=live)
    if camera.LIVE:
        camera.set_camera_settings(settings)
    print("choose background")
    bg = camera.background_and_wait()
    cv2.waitKey(0)  # wait to start game after background retrieval
    current = bg
    counter = 0
    buffer = []
    while camera.is_opened() and counter < 100:
        t1 = time.perf_counter()
        counter += 1
        current = camera.next_frame(current)
        temp_frame = current.copy()
        detection_results = Fd.fruit_detection(temp_frame, bg, CONTOUR_AREA_THRESH)
        cv2.drawContours(temp_frame, detection_results.conts, -1, (0, 255, 0), 2)
        # calculates meanshift for fruits known. removes fruits which left temp_frame.
        track_known_fruits(fruits_info, temp_frame, detection_results)
        if len(detection_results.conts) > 0:
            insert_new_fruits(detection_results, fruits_info, temp_frame)
        for fruit in fruits_info:
            if not fruit.is_falling:
                draw(fruit, temp_frame)
        # cv2.drawContours(temp_frame, detection_results.conts, -1, (0, 255, 0), 2)
        cv2.imshow("temp_frame", temp_frame)
        buffer.append(temp_frame)
        t2 = time.perf_counter()
        print("time for everything", abs(t1 - t2))
        if cv2.waitKey(1) == 27:
            break
        print("len of fruits: " + str(len(fruits_info)))

    # debug_with_buffer(buffer)
    show_original(camera)


def debug_with_buffer(buffer):
    i = 0
    while True:
        for fruit in fruits_for_debug_trajectories:
            draw_center(fruit, buffer[i])
            draw_trajectory(fruit, buffer[i])

        cv2.imshow("debug", buffer[i])
        x = cv2.waitKey(1)
        if x == 49:  # '1' key
            i -= 1
        elif x == 50:  # '2' key
            i += 1


def show_original(camera):
    i = 0
    while True:
        frame = camera.buffer[i]
        # frame = cv2.resize(frame, None, fx=0.3, fy=0.3)

        for fruit in fruits_for_debug_trajectories:
            # draw_center(fruit,frame)
            # if int(fruit.time_created * 30 ) < i and int(fruit.time_created * 30 + 120) > i:
            draw_trajectory(fruit, frame)


        frame = cv2.flip(frame, -1)
        cv2.putText(frame, 'SHANINJA', (240, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("debug", frame)
        x = cv2.waitKey(1)
        if x == 49:  # '1' key
            i -= 1
        elif x == 50:  # '2' key
            i += 1


def draw(fruit, frame):
    """
    auto generated function, does some things and displays shit.
    """
    draw_rectangles(fruit, frame, (255, 0, 0))
    draw_center(fruit, frame)  # conts_and_rects holds all the centers of all fruits - it has a list of lists.


if __name__ == '__main__':
    run_detection("saturdayDark.flv", Ci.DARK_101_SETTINGS, live=False, crop=True, flip=True)
