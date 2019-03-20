"""
TODO add documentation to all file
"""

import cv2
import time
import FruitDetection


def show_webcam(mirror=False):
    """
    TODO add documentation
    :param mirror:
    """
    cam = cv2.VideoCapture(0)
    # set_camera_settings(cam)
    ret_val, back = cam.read()
    while True:
        t1 = time.perf_counter()
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        cont_shit = FruitDetection.fruit_detection(img, back, 3000)
        cv2.drawContours(img, cont_shit[0], -1, (0, 255, 0), 2)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        t2 = time.perf_counter()
        # print("time for frame: " + str(t2 - t1))
    cv2.destroyAllWindows()


def set_camera_settings(cam):
    """
    TODO add documentation
    :param cam:
    """
    #       key value
    # cam.set(3, 1080)  # width
    # cam.set(4, 720)  # height
    # cam.set(10, 120)  # brightness      min: 0   , max: 255 , increment:1
    # cam.set(11, 120)  # contrast        min: 0   , max: 255 , increment:1
    # cam.set(12, 120)  # saturation      min: 0   , max: 255 , increment:1
    # cam.set(13, 13)  # hue
    cam.set(14, 120)  # gain              min: 0   , max: 127 , increment:1
    cam.set(15, -7)  # exposure           min: -7  , max: -1  , increment:1
    # cam.set(17, 5000)  # white_balance  min: 4000, max: 7000, increment:1
    cam.set(28, 13)  # focus              min: 0   , max: 255 , increment:5


if __name__ == '__main__':
    show_webcam(mirror=False)
