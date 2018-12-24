import cv2
import numpy as np

class VideoInterface:
    def __init__(self, src = 0):
        self.src = src

    def wait(self, x, cap):
        counter = 0
        while counter < 30 * x:
            cap.grab()
            counter += 1

    def crop_image(self, frame):
        (height, width, depth) = frame.shape
        new_h = int(height / 3)
        new_w = int(width / 7)
        frame = frame[2 * new_h:height, new_w:6 * new_w]
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        return frame

