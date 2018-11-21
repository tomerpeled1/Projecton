import cv2
import numpy as np
import time


def video_reader(name):
    cap = cv2.VideoCapture(name) #creates a video reading object.
    counter = 0
    cap.grab()
    original = cv2.imread("pic2.jpg") #retrieves the original frame.
    while True:
        if cap.grab() and counter >= 180 and counter%1 == 0:#grabs a frame. tempo can be controlled.
            flag, frame = cap.retrieve()#retrieves the frame into frame. flag = true if successful.
            if counter == 180:
                flag, original = cap.retrieve()
                original = crop_image(original)
            if not flag:
                continue
            else:
                t = time.clock()
                frame = crop_image(frame)
                frame = cv2.absdiff(frame,original)
                cv2.imshow("video", frame)
                #break
                print(time.clock() - t)
        if cv2.waitKey(10) == 27:
            break
        counter += 1

def crop_image(frame):
    (height, width, depth) = frame.shape
    new_w = int(width/3)
    new_h = int(height/3)
    return frame[2*new_h:height, :]

video_reader("first_video.mp4")