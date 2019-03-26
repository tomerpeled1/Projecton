"""
Implementation of fruit detection algorithm.
With a given frame, should return all fruit elements found in it.
"""

import numpy as np
import cv2
import time
from numpy import array

import scipy.signal
import matplotlib.pyplot as plt

import DetectionResults
from CameraInterface import Camera

pine_image = cv2.imread("pineapple.png")
pine_image = cv2.cvtColor(pine_image, cv2.COLOR_RGB2HSV)
PINEAPPLE_HIST = cv2.calcHist([pine_image], [0], None, [180], [1, 180])
# PINEAPPLE_HIST = [int(x) for x in PINEAPPLE_HIST]
# PINEAPPLE_HIST = PINEAPPLE_HIST[1:]
NORM_PINEAPPLE_HIST = cv2.normalize(PINEAPPLE_HIST, PINEAPPLE_HIST, norm_type=cv2.NORM_L1)

PINEAPPLE_THRESHOLD = 0.4
k=1

def fruit_detection2(frame, background, contour_area_thresh, time_of_frame):
    """
    fruit detection algorithm. based on background reduction and hsv color format
    :param frame: current frame to find fruits in
    :param background: according background to frame
    :param contour_area_thresh: minimal size of fruit
    :return: Detection result object - containing 3 list (every fruit has contour, surrounding rectangle and center)
    """
    t = time.perf_counter()

    current = frame
    back = background

    # split hvs of current frame
    current_h, current_v = get_hue_and_value(current)

    # split hvs of background
    back_h, back_v = get_hue_and_value(back)


    subtract_v = cv2.absdiff(current_v, back_v)
    subtract_and_thresh_v = thresh_value(subtract_v)
    # cv2.imshow("subtract_v", subtract_v)
    first_subtract_h = cv2.absdiff(current_h, back_h)
    # cv2.imshow("subtract_hue", first_subtract_h)
    subtract_h = clean_hue(first_subtract_h)
    # cv2.imshow("clean_hue", subtract_h)
    subtract_and_thresh_h = thresh_hue(current_h, subtract_h)
    # cv2.imshow("thresh_hue", subtract_and_thresh_h)
    or_image = cv2.bitwise_or(subtract_and_thresh_v, subtract_and_thresh_h)

    # calc total change (value + hue) and remove noise
    weighted_v = cv2.convertScaleAbs(subtract_v, alpha=0.75)
    weighted_h = cv2.convertScaleAbs(subtract_h, alpha=0.75)
    sub_add = cv2.add(weighted_v, subtract_h)
    # cv2.imshow("sub_add", sub_add)
    ret3, add_thresh = cv2.threshold(sub_add, 35, 255, cv2.THRESH_BINARY)
    add_thresh = cv2.morphologyEx(add_thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # remove noise


    # connect pieces of fruit and remove noises
    # mask = open_and_close(or_image)
    mask2 = open_and_close(add_thresh)  # todo maybe use add and choose appropriate threshold
    # cv2.imshow("or_image", mask)
    # cv2.imshow("open close", mask2)

    im2, cont, hier = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = [c for c in cont if cv2.contourArea(c) > contour_area_thresh]

    value_black = black_outside_contours(subtract_v, cont)
    hue_black = black_outside_contours(subtract_h, cont)

    # cv2.imshow("value_black", value_black)
    # cv2.imshow("hue_black", hue_black)

    new_cont = []
    for cnt in cont:
        x, y, w, h = cv2.boundingRect(cnt)
        value_window = second_threshold(value_black, x, y, w, h)
        # cv2.imshow("value_win", value_black)
        hue_window = second_threshold(hue_black, x, y, w, h)
        # cv2.imshow("hue_black", hue_black)
        # gray_window = second_threshold(gray_black, x, y, w, h)
        # cv2.imshow("gray_black", gray_black)
        # or_image = cv2.bitwise_or(value_window, gray_window)
        # cv2.imshow("or window", or_image)

        # sum_gray_and_hue = cv2.add(gray_black, value_black)
        # cv2.imshow("sum", sum_gray_and_hue)
        # sum_window = second_threshold(sum_gray_and_hue, x, y, w, h)

        # sub_gray_and_value_by_hue = cv2.subtract(or_image, hue_window)
        # cv2.imshow("sub", sub_gray_and_value_by_hue)

        im2, contour, hier = cv2.findContours(value_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len1 = len(new_cont)
        for c in contour:
            if cv2.contourArea(c) > contour_area_thresh:
                new_cont.append(move_back_contour(c, (x, y, w, h)))
        len2 = len(new_cont)
        if len1 == len2:
            if cv2.contourArea(cnt) > contour_area_thresh:
                new_cont.append(cnt)

    # calculate coordinates of surrounding rect of cont
    conts = []
    rects = []
    centers = []
    masked = cv2.bitwise_and(current, current, mask=mask2)
    # cv2.imshow("ma", masked)
    for i in range(len(cont)):

        c = cont[i]
        rect = extract_rect(c)
        center = center_of_contour(c)

        # show_cont(c, frame)

        # if False:
        #     pass
        if is_pineapple(masked, c, rect):
            conts.append(c)
            rects.append(rect)
            centers.append(center)

        else:
            new_conts, new_rects, new_centers = separate_overlap(masked, c, rect, center, contour_area_thresh)
            conts.extend(new_conts)
            rects.extend(new_rects)
            centers.extend(new_centers)

    # print("time for detection: " + str(time.perf_counter()-t))

    return DetectionResults.DetectionResults(conts, rects, centers,
                                             time_of_frame)  # list of lists, representing all fruits found


def is_pineapple(detection_frame, cont, rect):
    hist = get_hist(detection_frame, cont, rect)
    norm_hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    correlation = cv2.compareHist(norm_hist, NORM_PINEAPPLE_HIST, cv2.HISTCMP_CORREL)
    return correlation > PINEAPPLE_THRESHOLD

def get_hist(detection_frame, cont, rect):
    [bot_left, up_right] = rect
    (x_min, y_min) = bot_left
    (x_max, y_max) = up_right
    crop_by_rect = detection_frame[y_min:y_max, x_min:x_max]
    # if k == 2:
    #     cv2.imwrite("pineapple.png", crop_by_rect)
    crop_by_rect_hsv = cv2.cvtColor(crop_by_rect, cv2.COLOR_RGB2HSV)
    crop_hist = cv2.calcHist([crop_by_rect_hsv], [0], None, [180], [1, 180])
    # crop_hist = [int(x) for x in crop_hist]
    # crop_hist = crop_hist[1:]
    return crop_hist

def show_cont(name, cont, frame):
    x, y, w, h = cv2.boundingRect(cont)
    window = frame[y:y + h, x:x + w]
    cv2.imshow(name, window)

def move_back_contour(contour, original_rect):
    x, y, w, h = original_rect
    for point in contour:
        point[0][0] += x
        point[0][1] += y
    return contour


def get_hue_and_value(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    current_h, _, current_v = cv2.split(hsv_frame)
    current_h = cv2.convertScaleAbs(current_h, alpha=255 / 179)  # converts hue to full spectrum
    current_h = cv2.morphologyEx(current_h, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # noise removal
    return current_h, current_v


def thresh_value(subtract_v):
    ret3, add_thresh = cv2.threshold(subtract_v, 30, 255, cv2.THRESH_BINARY)
    add_thresh = cv2.morphologyEx(add_thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # remove noise
    det1 = add_thresh
    return det1


def thresh_hue(current_h, subtract_h):
    white_img = 255 * np.ones(current_h.shape, np.uint8)
    complement_subtract_h = cv2.subtract(white_img, subtract_h)  # second cyclic option
    final_sub_h = cv2.min(subtract_h, complement_subtract_h)  # modification to cyclic scaling
    subtract_h_mod = cv2.convertScaleAbs(final_sub_h, alpha=1.3)  # amplify hue

    ret3, add_thresh = cv2.threshold(subtract_h_mod, 30, 255, cv2.THRESH_BINARY)
    add_thresh = cv2.morphologyEx(add_thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # remove noise
    det2 = add_thresh
    return det2


def open_and_close(frame):
    ellipse1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, ellipse1)
    ellipse2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ellipse2)
    return opened


def black_outside_contours(frame, conts):
    stencil = np.zeros(frame.shape).astype(frame.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, conts, color)
    res = cv2.bitwise_and(frame, stencil)
    return res


def second_threshold(frame, x, y, w, h):
    window = frame[y:y + h, x:x + w]
    norm_image = cv2.normalize(window, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow("norm", norm_image)
    thresh = 100
    ret, th = cv2.threshold(norm_image, thresh, 255, cv2.THRESH_BINARY)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    close = cv2.morphologyEx(th, cv2.MORPH_CLOSE, ellipse)
    return close


def clean_hue(subtract_h):
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(subtract_h, cv2.MORPH_OPEN, ellipse)
    ret, thr = cv2.threshold(opened, 30, 255, cv2.THRESH_BINARY)
    cleaned = cv2.bitwise_and(subtract_h, thr)
    return cleaned


# def otsu(gray):
#     pixel_number = gray.shape[0] * gray.shape[1]
#     mean_weigth = 1.0/pixel_number
#     his, bins = np.histogram(gray, np.array(range(0, 256)))
#     final_thresh = -1
#     final_value = -1
#     otsu_factor = mean_weigth
#     for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
#         Wb = np.sum(his[:t]) * otsu_factor
#         Wf = np.sum(his[t:]) * otsu_factor
#
#         mub = np.mean(his[:t])
#         muf = np.mean(his[t:])
#
#         value = Wb * Wf * (mub - muf) ** 2
#
#         print("Wb", Wb, "Wf", Wf)
#         print("t", t, "value", value)
#
#         if value > final_value:
#             final_thresh = t
#             final_value = value
#     return final_thresh


def center_of_contour(c):
    """
    given contour, calc its center of mass (opencv standard)
    :param c: contour object
    :return: center of mass in pixels (height, width)
    """
    m = cv2.moments(c)
    y = m["m00"]
    if y == 0:
        y = 0.000001
    c_x = int(m["m10"] / y)
    c_y = int(m["m01"] / y)
    return c_x, c_y


def separate_overlap(detection_frame, cont, rect, cent, cont_area_thresh):
    # global k
    crop_hist = get_hist(detection_frame, cont, rect)
    # print(str(k))
    # print(crop_hist)
    # show_cont(str(k), cont, detection_frame)
    # k+=1
    crop_hist = [int(x) for x in crop_hist]
    crop_hist = crop_hist[1:]
    sample_sum = sum(crop_hist)
    crop_hist[:] = [col / sample_sum for col in crop_hist]

    # cv2.imshow("a", crop_by_rect)
    # cv2.waitKey(0)
    # plt.bar(np.arange(1,180), crop_hist, align='center', alpha=0.5)
    # plt.show()


    main_colors = analyze_hist_to_colors(crop_hist, cont_area_thresh, sample_sum)

    conts = []
    rects = []
    cents = []

    color_variance = 15
    if len(main_colors) == 1:
        return [cont], [rect], [cent]

    [bot_left, up_right] = rect
    for color in main_colors:
        area_mask = cv2.rectangle(np.zeros(detection_frame.shape, np.uint8), bot_left, up_right, (255, 255, 255), -1)
        masked_area = cv2.bitwise_and(detection_frame, area_mask)
        low_h = max(1, color - color_variance)
        high_h = min(179, color + color_variance)
        low_hsv = np.array([low_h, 0, 0])
        high_hsv = np.array([high_h, 255, 255])
        masked_area_hsv = cv2.cvtColor(masked_area, cv2.COLOR_RGB2HSV)
        masked_color = cv2.inRange(masked_area_hsv, low_hsv, high_hsv)
        masked_color = cv2.morphologyEx(masked_color, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        im2, contours, hier = cv2.findContours(masked_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("masked color", masked_color)
        thresh_contours = [c for c in contours if cv2.contourArea(c) > cont_area_thresh]

        # if(len(thresh_contours) > 1):
        #     print("this is a problem, too many contours: " + str(len(thresh_contours)))
        if len(thresh_contours) != 0:
            conts.extend(thresh_contours)
            rects.extend([extract_rect(c) for c in thresh_contours])
            cents.extend([center_of_contour(c) for c in thresh_contours])

    def cont_area(tup):
        cnt, rct, cent = tup
        return cv2.contourArea(cnt)

    lst = [(conts[i], rects[i], cents[i]) for i in range(len(conts))]

    lst.sort(key=cont_area)

    lst_indices_to_remove = []
    for i in range(len(conts)):
        cont1 = lst[i][0]
        for j in range(i):
            cont2 = lst[j][0]
            if is_cont1_contained_by_cont2(cont1, cont2, detection_frame):
                if j not in lst_indices_to_remove:
                    lst_indices_to_remove.append(j)

    lst_indices_to_remove.sort(reverse=True)
    for i in lst_indices_to_remove:
        lst.pop(i)

    # keeps the given contour if it does not divide it to at least 2 contours
    if len(lst) < 2:
        return [cont], [rect], [cent]

    conts = [tup[0] for tup in lst]
    rects = [tup[1] for tup in lst]
    cents = [tup[2] for tup in lst]

    return conts, rects, cents


def find_biggest_contour(conts):
    areas = [cv2.contourArea(c) for c in conts]
    largest_size = max(areas)
    index = areas.index(largest_size)
    return conts[index]


def analyze_hist_to_colors(hist, min_contour_area, num_of_pix):
    color_variance = 8
    norm_hist = []
    for i in range(0, 179):
        low = max(0, i - color_variance)
        high = min(178, i + color_variance) + 1
        norm_hist.append(sum(hist[low:high]))
    # plt.bar(np.arange(1, 180), norm_hist, align='center', alpha=0.5)
    # plt.show()
    peaks, _ = scipy.signal.find_peaks(norm_hist)
    return [peak for peak in peaks if norm_hist[peak] >= min_contour_area / num_of_pix]


def extract_rect(c):
    x_min = c[c[:, :, 0].argmin()][0][0]
    x_max = c[c[:, :, 0].argmax()][0][0]
    y_min = c[c[:, :, 1].argmin()][0][1]
    y_max = c[c[:, :, 1].argmax()][0][1]
    bot_left = (x_min, y_min)
    # up_left = (x_min, y_max)
    # bot_right = (x_max, y_min)
    up_right = (x_max, y_max)
    rect = [bot_left, up_right]
    return rect


def is_cont1_contained_by_cont2(cont1, cont2, frame):
    overlap_threshold = 0.5
    height, width, _ = frame.shape
    cont1_filled = np.zeros((height, width))
    cont2_filled = np.zeros((height, width))
    cv2.fillPoly(cont1_filled, pts=[cont1], color=(255, 255, 255))
    cv2.fillPoly(cont2_filled, pts=[cont2], color=(255, 255, 255))

    and_image = cv2.bitwise_and(cont1_filled, cont2_filled)

    and_pixels = cv2.countNonZero(and_image)
    cont1_pixels = cv2.countNonZero(cont1_filled)

    ratio = and_pixels / cont1_pixels

    if ratio > overlap_threshold:
        return True
    return False



if __name__ == "__main__":
    resize_factor = 0.3
    minimal_contour_area = 4000
    cap = Camera("img&vid\\23-3Dark.flv", flip=True, crop=False, live=False)
    cnt = 0
    while (cnt < 1):
        back_main = cap.read()[0]
        cnt += 1
    cv2.imshow("main", back_main)
    cv2.waitKey(0)
    back_main = cv2.resize(back_main, None, fx=resize_factor, fy=resize_factor)
    cv2.imshow("main", back_main)
    cv2.waitKey(0)
    first_time = time.time()
    while (cap.is_opened()):
        frame_main = cap.read()[0]
        frame_main = cv2.resize(frame_main, None, fx=resize_factor, fy=resize_factor)
        (height, width, depth) = frame_main.shape
        # back_main = cap.read()
        # back_main = cv2.resize(back_main, None, fx=0.3, fy=0.3)

        detection_results = fruit_detection2(frame_main, back_main, resize_factor**2*minimal_contour_area, time.time() - first_time)
        cv2.drawContours(frame_main, detection_results.conts, -1, (0, 255, 0), 2)
        # for i in range(len(rects)):
        #     frame = cv2.rectangle(frame, rects[i][UP_LEFT], rects[i][BOTTOM_RIGHT],
        #                       (255, 0, 0), 2)
        cv2.imshow("frame", frame_main)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


    # resize_factor = 0.3
    # minimal_contour_area = 4000
    # # cap = Camera("img&vid\\23-3Dark.flv", flip=True, crop=False, live=False)
    # cnt = 0
    # # while (cnt < 1):
    # #     back_main = cap.read()[0]
    # #     cnt += 1
    # back_main = cv2.imread("img&vid\\pineBack.png")
    # cv2.imshow("main", back_main)
    # cv2.waitKey(0)
    # back_main = cv2.resize(back_main, None, fx=resize_factor, fy=resize_factor)
    # cv2.imshow("main", back_main)
    # cv2.waitKey(0)
    # first_time = time.time()
    # if True:
    # # while (cap.is_opened()):
    # #     frame_main = cap.read()[0]
    #     frame_main = cv2.imread("img&vid\\pineFront.png")
    #     frame_main = cv2.resize(frame_main, None, fx=resize_factor, fy=resize_factor)
    #     (height, width, depth) = frame_main.shape
    #     # back_main = cap.read()
    #     # back_main = cv2.resize(back_main, None, fx=0.3, fy=0.3)
    #
    #     detection_results = fruit_detection2(frame_main, back_main, resize_factor**2*minimal_contour_area, time.time() - first_time)
    #     cv2.drawContours(frame_main, detection_results.conts, -1, (0, 255, 0), 2)
    #     # for i in range(len(rects)):
    #     #     frame = cv2.rectangle(frame, rects[i][UP_LEFT], rects[i][BOTTOM_RIGHT],
    #     #                       (255, 0, 0), 2)
    #     cv2.imshow("frame", frame_main)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
