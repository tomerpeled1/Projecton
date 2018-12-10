import cv2
import numpy as np
import time
import FruitDetection
# import TrackerIntegrate as ti

CONT = 0
RECT = 1
CENTER = 2


def video_reader(name):
    cropped = False
    cap = cv2.VideoCapture(name) #creates a video reading object.
    counter = 0
    wait(6, cap)
    original = get_background(cap)
    while True:
        if cap.grab() and counter > 300 and counter%1 == 0:#grabs a frame. tempo can be controlled.
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                frame = crop_image(frame)
                conts_and_rects = FruitDetection.fruit_detection(frame, original, 4100)
                cv2.drawContours(frame, conts_and_rects[CONT], -1, (0, 255, 0), 2)
                cv2.imshow("video", frame)
                cv2.waitKey(0)
            if cv2.waitKey(10) == 27:
                break
        counter += 1
    return

def crop_image(frame):
    (height, width, depth) = frame.shape
    new_h = int(height/3)
    new_w = int(width/7)
    frame = frame[2*new_h:height, new_w:6*new_w]
    frame = cv2.resize(frame,(0,0),fx = 0.7, fy = 0.7)
    return frame

def wait(x, cap):
    counter = 0
    while counter < 30*x:
        cap.grab()

        counter += 1

def get_background(cap):
    cap.grab()
    flag, frame = cap.retrieve()
    frame = crop_image(frame)
    if flag:
        return frame
    else:
        exit()


if __name__ == "__main__":
    video_reader("first_video.mp4")