import cv2
import numpy as np
import time


def video_reader(name):
    cap = cv2.VideoCapture(name) #creates a video reading object.
    counter = 0
    cap.grab()
    flag, original = cap.retrieve() #retrieves the original frame.
    while True:
        if cap.grab() and counter%1 == 0: #grabs a frame. tempo can be controlled.
            flag, frame = cap.retrieve() #retrieves the frame into frame. flag = true if successful.
            if not flag:
                continue
            else:
                t = time.clock()
                frame = frame - original
                cv2.imshow("video", frame)
                #cv2.waitKey(0)
                #break
                print(time.clock() - t)
        if cv2.waitKey(10) == 27:
            break
        counter += 1


video_reader("first_video.mp4")