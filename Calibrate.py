"""
Takes care of calibrating the location of the screen.
"""

import cv2


def get_bounding_rect(c):
    """
    Finds the bounding rectangle of a contour c.
    :param c: Contour to bound
    :return: The bounding rectangle in the format [bottom_left, top_right] when [0, 0] is bottom left corner of frame.
    """
    x_min = c[c[:, :, 0].argmin()][0][0]
    x_max = c[c[:, :, 0].argmax()][0][0]
    y_min = c[c[:, :, 1].argmin()][0][1]
    y_max = c[c[:, :, 1].argmax()][0][1]
    bot_left = (x_min, y_max)
    up_right = (x_max, y_min)
    rect = [bot_left, up_right]
    return rect


def calibrate(frame):
    """
    :param frame: A frame of white image on the tablet and everything else is normal
    :return: A rectangle in the regular format, where the bounds of the screens are.
    """
    im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", im_gray)
    cv2.waitKey(0)
    _, thresh = cv2.threshold(im_gray, 200, 255, 0)
    _, contours, __ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    white = []
    for c in contours:
        # Checks if the contour is large enough to be screen size.
        if cv2.contourArea(c) > 50000:
            white.append(c)
    rect = get_bounding_rect(white[0])
    cv2.drawContours(frame, white, -1, (0, 255, 0), 2)
    cv2.imshow("calibrated", frame)
    cv2.waitKey(0)
    return rect
