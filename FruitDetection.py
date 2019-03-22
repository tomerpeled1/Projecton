"""
Implementation of fruit detection algorithm.
With a given frame, should return all fruit elements found in it.
"""

import numpy as np
import scipy.signal
import cv2
import time
import DetectionResults
import matplotlib.pyplot as plt


def fruit_detection(frame, background, contour_area_thresh):
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
    current_hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    current_h, current_s, current_v = cv2.split(current_hsv)
    current_h = cv2.convertScaleAbs(current_h, alpha=255/179)  # converts hue to full spectrum
    current_h = cv2.morphologyEx(current_h, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # noise removal
    # cv2.imshow("real_h_mod", real_h)

    # split hvs of background
    back_hsv = cv2.cvtColor(back, cv2.COLOR_BGR2HSV)
    back_h, _, back_v = cv2.split(back_hsv)
    back_h = cv2.convertScaleAbs(back_h, alpha=255/179)  # converts hue to full spectrum
    back_h = cv2.morphologyEx(back_h, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # noise removal

    # find value change
    # print(real_v.shape, back_v.shape)
    subtract_v = cv2.absdiff(current_v, back_v)
    # cv2.imshow("sub_v", subtract_v)

    # find hue change (with attention to cyclic scaling), amplify hue
    subtract_h = cv2.absdiff(current_h, back_h)  # first cyclic option
    # cv2.imshow("sub_h_bef", subtract_h)
    white_img = 255*np.ones(current_h.shape, np.uint8)
    complement_subtract_h = cv2.subtract(white_img, subtract_h)  # second cyclic option
    final_sub_h = cv2.min(subtract_h, complement_subtract_h)  # modification to cyclic scaling
    subtract_h_mod = cv2.convertScaleAbs(final_sub_h, alpha=1.3)  # amplify hue
    # cv2.imshow("sub_h", subtract_h_mod)

    # calc total change (value + hue) and remove noise
    sub_add = cv2.add(subtract_v, subtract_h_mod)
    # cv2.imshow("sub_add", sub_add)
    ret3, add_thresh = cv2.threshold(sub_add, 80, 255, cv2.THRESH_BINARY)
    add_thresh = cv2.morphologyEx(add_thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # remove noise

    mask = add_thresh

    # connect pieces of fruit and remove noises
    # cv2.imshow("tt1", mask)
    y = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    z = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, z)
    # cv2.imshow("tt2", mask)
    y2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    z2 = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, z2)
    # cv2.imshow("tt3", mask)



    # apply mask
    masked = cv2.bitwise_and(current, current, mask=mask)
    # cv2.imshow("masked", masked)
    # cv2.waitKey(0)

    # create contours
    gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    im2, cont, hier = cv2.findContours(gray_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = [c for c in cont if cv2.contourArea(c) > contour_area_thresh]
    if len(cont) !=0:
          a=0

    copy = frame.copy()
    stencil = np.zeros(copy.shape).astype(copy.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, cont, color)
    res = cv2.bitwise_and(copy, stencil)
    # cv2.imshow("res", res)

    new_cont = []
    for cnt in cont:
        x, y, w, h = cv2.boundingRect(cnt)
        work = res.copy()
        to_show = work[y:y + h, x:x + w]
        gry = cv2.cvtColor(to_show, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gry, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        z = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        close = cv2.morphologyEx(th, cv2.MORPH_CLOSE, z)
        im2, contour, hier = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contour:
            new_cont.append(move_back_contour(c, (x,y,w,h)))

        # z = np.ones((5, 5), np.uint8)
        # close = cv2.morphologyEx(th, cv2.MORPH_CLOSE, z)
        #
        # cv2.imshow("th", th)


    # im2, cont, hier = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_cont = [c for c in new_cont if cv2.contourArea(c) > contour_area_thresh]

    # # try to improve detection and remove spreading
    # for cnt in cont:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     work = frame.copy()
    #
    #     cv2.drawContours(work, cnt, -1, (0, 255, 0), 2)
    #
    #     to_show = work[y:y + h, x:x + w]
    #
    #     to_show = cv2.cvtColor(to_show, cv2.COLOR_BGR2GRAY)
    #
    #     to_show = cv2.resize(to_show,(0,0), fx=3, fy=3)
    #     cv2.imshow("to_show", to_show)
    #     cv2.waitKey(0)

    # calculate coordinates of surrounding rect of cont
    conts = []
    rects = []
    centers = []
    for i in range(len(new_cont)):
        c = new_cont[i]
        rect = extract_rect(c)
        center = center_of_contour(c)

        new_conts, new_rects, new_centers = separate_overlap(masked, c, rect, center, contour_area_thresh)
        conts.extend(new_conts)
        rects.extend(new_rects)
        centers.extend(new_centers)


    print("time for detection: " + str(time.perf_counter()-t))

    return DetectionResults.DetectionResults(conts, rects, centers)  # list of lists, representing all fruits found

def separate_overlap(detection_frame, cont, rect, cent, cont_area_thresh):
    [bot_left, up_right] = rect
    (x_min, y_min) = bot_left
    (x_max, y_max) = up_right
    crop_by_rect = detection_frame[y_min:y_max, x_min:x_max]
    crop_by_rect_hsv = cv2.cvtColor(crop_by_rect, cv2.COLOR_RGB2HSV)
    crop_hist = cv2.calcHist([crop_by_rect_hsv], [0], None, [180],[0, 180])
    crop_hist = [int(x) for x in crop_hist]
    crop_hist = crop_hist[1:]
    sample_sum = sum(crop_hist)
    crop_hist[:] = [col / sample_sum for col in crop_hist]
    # cv2.imshow("a", crop_by_rect)
    # cv2.waitKey(0)
    # plt.bar(np.arange(1,180), crop_hist, align='center', alpha=0.5)
    # plt.show()
    main_colors = analyze_hist_to_colors(crop_hist)

    conts = []
    rects = []
    cents = []

    color_variance = 10
    if len(main_colors) == 1:
        return [cont], [rect], [cent]

    for color in main_colors:
        area_mask = cv2.rectangle(np.zeros(detection_frame.shape, np.uint8), bot_left, up_right, (255, 255, 255), -1)
        masked_area = cv2.bitwise_and(detection_frame, area_mask)
        low_h = max(0, color - color_variance)
        high_h = min(180, color + color_variance)
        low_hsv = np.array([low_h, 0, 0])
        high_hsv = np.array([high_h, 255, 255])
        masked_area_hsv = cv2.cvtColor(masked_area, cv2.COLOR_RGB2HSV)
        masked_color = cv2.inRange(masked_area_hsv, low_hsv, high_hsv)
        masked_color = cv2.morphologyEx(masked_color, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        im2, contours, hier = cv2.findContours(masked_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > cont_area_thresh]
        if(len(contours) > 1):
            print("this is a problem, too many contours")
        c = find_biggest_contour(contours)
        rect = extract_rect(c)
        center = center_of_contour(c)
        conts.append(c)
        rects.append(rect)
        cents.append(center)

    return conts, rects, cents

def find_biggest_contour(conts):
    areas = [cv2.contourArea(c) for c in conts]
    largest_size = max(areas)
    index = areas.index(largest_size)
    return conts[index]

def analyze_hist_to_colors(hist):
    color_variance = 5
    norm_hist = []
    for i in range(0, 179):
        low = max(0, i - color_variance)
        high = min(178, i + color_variance) + 1
        norm_hist.append(sum(hist[low:high]))
    # plt.bar(np.arange(1, 180), norm_hist, align='center', alpha=0.5)
    # plt.show()
    peaks, _ = scipy.signal.find_peaks(norm_hist)
    PERCENTAGE = 0.25
    return [peak for peak in peaks if norm_hist[peak] >= PERCENTAGE]


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

def move_back_contour(contour, original_rect):
    x,y,w,h = original_rect
    for point in contour:
        point[0][0] += x
        point[0][1] += y
    return contour

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
    frame_main = cv2.imread("img&vid\greyWatermelon.png")
    # frame_main = cv2.resize(frame_main, None, fx=0.3, fy=0.3)
    (height, width, depth) = frame_main.shape
    back_main = cv2.imread("img&vid\grayBack.png")
    # back_main = cv2.resize(back_main, None, fx=0.3, fy=0.3)

    dr = fruit_detection(frame_main, back_main, 1000)
    cv2.drawContours(frame_main, dr.conts, -1, (0, 255, 0), 2)
    # for i in range(len(rects)):
    #     frame = cv2.rectangle(frame, rects[i][UP_LEFT], rects[i][BOTTOM_RIGHT],
    #                       (255, 0, 0), 2)
    cv2.imshow("frame", frame_main)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
