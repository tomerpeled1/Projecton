import cv2
import numpy as np

def get_bounding_rect(c):
    x_min = c[c[:, :, 0].argmin()][0][0]
    x_max = c[c[:, :, 0].argmax()][0][0]
    y_min = c[c[:, :, 1].argmin()][0][1]
    y_max = c[c[:, :, 1].argmax()][0][1]
    bot_left = (x_min, y_min)
    up_left = (x_min, y_max)
    bot_right = (x_max, y_min)
    up_right = (x_max, y_max)
    rect = [bot_left, up_right]
    return rect


def calibrate(frame):

    '''
    :param frame: a frame of white image on the tablet and everything else is normal
    :return: a rectangle in the regular format, where the boundries of the screens are.
    '''
    imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(imgray,100,255,0)
    _, contours, __ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    white = []
    for c in contours:
        if cv2.contourArea(c) > 5000:
            white.append(c)
    rect = get_bounding_rect(white[0])
    return rect




if __name__ == '__main__':
    cal = cv2.imread("calibrate_image.jpg")
    cal = cv2.resize(cal,(0,0),fx = 0.5, fy = 0.5)

    calibrate(cal)