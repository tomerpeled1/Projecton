import numpy as np
import cv2
import time

def fruit_detection(frame, background, contour_area_thresh):
    t = time.perf_counter()

    real = frame
    back = background

    # split hvs of frame
    real_hsv = cv2.cvtColor(real, cv2.COLOR_BGR2HSV)
    real_h, real_s, real_v = cv2.split(real_hsv)
    real_h = cv2.convertScaleAbs(real_h, alpha=255/179)
    real_h = cv2.morphologyEx(real_h, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # split hvs of background
    back_hsv = cv2.cvtColor(back, cv2.COLOR_BGR2HSV)
    back_h, _, back_v = cv2.split(back_hsv)
    back_h = cv2.convertScaleAbs(back_h, alpha=255/179)
    back_h = cv2.morphologyEx(back_h, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # find value change
    subtract_v = cv2.absdiff(real_v, back_v)

    # find hue change, amplify hue
    subtract_h = cv2.absdiff(real_h, back_h)
    subtract_h_mod = cv2.convertScaleAbs(subtract_h, alpha=1.3)

    # calc total change (value + hue) and remove noise
    sub_add = cv2.add(subtract_v, subtract_h_mod)
    ret3, add_thresh = cv2.threshold(sub_add, 55, 255, cv2.THRESH_BINARY)
    add_thresh = cv2.morphologyEx(add_thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # mask that removes very bright noises (blade)
    ret, mask_s = cv2.threshold(real_s, 31, 255, cv2.THRESH_BINARY)

    # combine masks
    mask = cv2.bitwise_and(add_thresh, mask_s)

    #connect pieces of fruit and remove noises
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((10,10), np.uint8))

    # apply mask
    masked = cv2.bitwise_and(real, real, mask=mask)

    # find lapping fruit - not ready!!!
    masked_hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    masked_h = masked_hsv[:, :, 0]
    #cv2.imshow("masked_h", masked_h)
    blurred_masked_h = cv2.blur(masked_h, (5, 5))
    #cv2.imshow("b_masked_h", blurred_masked_h)
    normalized_masked_h = cv2.normalize(blurred_masked_h, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    #cv2.imshow("norm", normalized_masked_h)
    gradient_hue = cv2.morphologyEx(normalized_masked_h, cv2.MORPH_GRADIENT, None)
    # cv2.imshow("gradient_hue", gradient_hue)

    # create contours
    gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    im2, cont, hier = cv2.findContours(gray_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = [c for c in cont if cv2.contourArea(c) > contour_area_thresh]

    # calculate coordinates of surrounding rect of cont
    cont_rect_coordinates = []
    for c in cont:
        x_min = c[c[:, :, 0].argmin()][0][0]
        x_max = c[c[:, :, 0].argmax()][0][0]
        y_min = c[c[:, :, 1].argmin()][0][1]
        y_max = c[c[:, :, 1].argmax()][0][1]
        bot_left = (x_min, y_min)
        up_left = (x_min, y_max)
        bot_right = (x_max, y_min)
        up_right = (x_max, y_max)
        rect = [bot_left, up_left, bot_right, up_right]
        cont_rect_coordinates.append(rect)

    print(time.perf_counter()-t)

    return cont, cont_rect_coordinates

if __name__ == "__main__":
    frame = cv2.imread("pic3.jpg")
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    (height, width, depth) = frame.shape
    back = cv2.imread("pic2.jpg")
    back = cv2.resize(back, None, fx=0.3, fy=0.3)

    cont, rects = fruit_detection(frame, back, 1000)[0]
    cv2.drawContours(frame, cont, -1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()