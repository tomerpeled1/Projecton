"""
Implementation of fruit detection algorithm.
With a given frame, should return all fruit elements found in it.
"""

import numpy as np
import cv2
import time
import DetectionResults
from CameraInterface import Camera


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
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    # split hvs of current frame
    current_h, current_v = get_hue_and_value(current)

    # split hvs of background
    back_h, back_v = get_hue_and_value(back)

    subtract_v = cv2.absdiff(current_v, back_v)
    subtract_and_thresh_v = thresh_value(subtract_v)
    subtract_h = cv2.absdiff(current_h, back_h)
    subtract_h = clean_hue(subtract_h)
    # cv2.imshow("hue", subtract_h)
    subtract_and_thresh_h = thresh_hue(current_h, subtract_h)
    or_image = cv2.bitwise_or(subtract_and_thresh_v, subtract_and_thresh_h)

    # calc total change (value + hue) and remove noise
    mask = or_image

    # connect pieces of fruit and remove noises
    mask = open_and_close(mask)
    # cv2.imshow("open close", mask)
    # apply mask

    v_and_h = cv2.bitwise_or(subtract_v, subtract_h)
    # cv2.imshow("v and h", v_and_h)
    masked = cv2.bitwise_and(v_and_h, v_and_h, mask=mask)
    masked = cv2.bitwise_and(current, current, mask=mask)

    # create contours
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", masked)
    ret, thresh = cv2.threshold(masked, 30, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    open_close = open_and_close(thresh)

    # cv2.imshow("before first contour", open_close)

    im2, cont, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = [c for c in cont if cv2.contourArea(c) > contour_area_thresh]

    value_black = black_outside_contours(subtract_v, cont)
    hue_black = black_outside_contours(subtract_h, cont)
    gray_black = black_outside_contours(gray, cont)
    # cv2.imshow("value_black", value_black)
    # cv2.imshow("res", res)

    new_cont = []
    for cnt in cont:
        x, y, w, h = cv2.boundingRect(cnt)
        value_window = second_threshold(value_black, x, y, w, h)
        # cv2.imshow("value_black", value_black)
        hue_window = second_threshold(hue_black, x, y, w, h)
        # cv2.imshow("hue_black", hue_window)
        gray_window = second_threshold(gray_black, x, y, w, h)
        # cv2.imshow("gray_black", gray_black)
        or_image = cv2.bitwise_or(value_window, gray_window)
        # cv2.imshow("or window", or_image)

        sum_gray_and_hue = cv2.add(gray_black, value_black)
        # cv2.imshow("sum", sum_gray_and_hue)
        sum_window = second_threshold(sum_gray_and_hue, x, y, w, h)

        # sub_gray_and_value_by_hue = cv2.subtract(or_image, hue_window)
        # cv2.imshow("sub", sub_gray_and_value_by_hue)

        im2, contour, hier = cv2.findContours(sum_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    for i in range(len(new_cont)):
        c = new_cont[i]
        x_min = c[c[:, :, 0].argmin()][0][0]
        x_max = c[c[:, :, 0].argmax()][0][0]
        y_min = c[c[:, :, 1].argmin()][0][1]
        y_max = c[c[:, :, 1].argmax()][0][1]
        bot_left = (x_min, y_min)
        # up_left = (x_min, y_max)
        # bot_right = (x_max, y_min)
        up_right = (x_max, y_max)
        rect = [bot_left, up_right]
        center = center_of_contour(c)
        conts.append(c)
        rects.append(rect)
        centers.append(center)

    # print("time for detection: " + str(time.perf_counter()-t))

    return DetectionResults.DetectionResults(conts, rects, centers,
                                             time_of_frame)  # list of lists, representing all fruits found


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
    thresh = 160
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


if __name__ == "__main__":
    cap = Camera("2019-03-17 19-59-34.flv", flip=False, crop=False, live=False)
    cnt = 0
    while (cnt < 23):
        back_main = cap.read()[0]
        cnt += 1
    cv2.imshow("main", back_main)
    cv2.waitKey(0)
    back_main = cv2.resize(back_main, None, fx=0.5, fy=0.5)
    cv2.imshow("main", back_main)
    cv2.waitKey(0)
    first_time = time.time()
    while (cap.is_opened()):
        frame_main = cap.read()[0]
        frame_main = cv2.resize(frame_main, None, fx=0.5, fy=0.5)
        (height, width, depth) = frame_main.shape
        # back_main = cap.read()
        # back_main = cv2.resize(back_main, None, fx=0.3, fy=0.3)

        detection_results = fruit_detection2(frame_main, back_main, 700, time.time() - first_time)
        cv2.drawContours(frame_main, detection_results.conts, -1, (0, 255, 0), 2)
        # for i in range(len(rects)):
        #     frame = cv2.rectangle(frame, rects[i][UP_LEFT], rects[i][BOTTOM_RIGHT],
        #                       (255, 0, 0), 2)
        cv2.imshow("frame", frame_main)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
