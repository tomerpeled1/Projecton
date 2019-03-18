"""
Implementation of fruit detection algorithm.
With a given frame, should return all fruit elements found in it.
"""

import numpy as np
import cv2
import time
import DetectionResults


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

    # mask that removes very bright noises (blade)
    # ret, mask_s = cv2.threshold(real_s, 31, 255, cv2.THRESH_BINARY)

    # combine masks
    # mask = cv2.bitwise_and(add_thresh, mask_s)
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

    # find lapping fruit - not ready!!!
    # masked_hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    # masked_h = masked_hsv[:, :, 0]
    # cv2.imshow("masked_h", masked_h)
    # blurred_masked_h = cv2.blur(masked_h, (5, 5))
    # cv2.imshow("b_masked_h", blurred_masked_h)
    # normalized_masked_h = cv2.normalize(blurred_masked_h, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    # cv2.imshow("norm", normalized_masked_h)
    # gradient_hue = cv2.morphologyEx(normalized_masked_h, cv2.MORPH_GRADIENT, None)
    # cv2.imshow("gradient_hue", gradient_hue)

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

    return DetectionResults.DetectionResults(conts, rects, centers)  # list of lists, representing all fruits found


def move_back_contour(contour, original_rect):
    x,y,w,h = original_rect
    for point in contour:
        point[0][0] += x
        point[0]  [1] += y
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
    frame_main = cv2.imread("pic1.jpg")
    frame_main = cv2.resize(frame_main, None, fx=0.3, fy=0.3)
    (height, width, depth) = frame_main.shape
    back_main = cv2.imread("pic2.jpg")
    back_main = cv2.resize(back_main, None, fx=0.3, fy=0.3)

    conts_main, rects_main = fruit_detection(frame_main, back_main, 1000)
    cv2.drawContours(frame_main, conts_main, -1, (0, 255, 0), 2)
    # for i in range(len(rects)):
    #     frame = cv2.rectangle(frame, rects[i][UP_LEFT], rects[i][BOTTOM_RIGHT],
    #                       (255, 0, 0), 2)
    cv2.imshow("frame", frame_main)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
