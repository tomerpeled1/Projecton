import cv2
from imutils.video import WebcamVideoStream
import time


def stream_cam(src):
    stream = WebcamVideoStream(src).start()
    ##set_camera_settings(stream.stream)
    counter = 0
    current = stream.read()
    frames = [current]

    while counter < 300:
        t1 = time.perf_counter()
        next = next_frame(current, stream)
        frames.append(next)
        current = next
        cv2.imshow("LIVE", current)
        if cv2.waitKey(1) == 27:
            break
        t2 = time.perf_counter()
        print("time for frame: " + str(t2 - t1))
        counter += 1
    for frame in frames:
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    stream.stop()

def next_frame(current, stream):
    while True:
        to_return = stream.read()
        dif = cv2.subtract(to_return, current)
        dif = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
        if (cv2.countNonZero(dif) > 0):
            return to_return



def set_camera_settings(cam):
    #       key value
    # cam.set(3, 1920)  # width
    # cam.set(4, 1080)  # height
    cam.set(10, 120)  # brightness     min: 0   , max: 255 , increment:1
    cam.set(11, 120)  # contrast       min: 0   , max: 255 , increment:1
    cam.set(12, 120)  # saturation     min: 0   , max: 255 , increment:1
    cam.set(13, 13)  # hue
    cam.set(14, 50)  # gain           min: 0   , max: 127 , increment:1
    cam.set(15, -6)  # exposure       min: -7  , max: -1  , increment:1
    cam.set(17, 5000)  # white_balance  min: 4000, max: 7000, increment:1
    cam.set(28, 0)  # focus          min: 0   , max: 255 , increment:5

def crop_image(frame):
    (height, width, depth) = frame.shape
    new_h = int(height/3)
    new_w = int(width/7)
    frame = frame[2*new_h:height, new_w:6*new_w]
    frame = cv2.resize(frame,(0,0),fx = 0.5, fy = 0.5)
    return frame

def wait(self, x, stream):
    cur = stream.read()
    counter = 0
    while counter < 30 * x:
        cur = next_frame(cur, stream)
        counter += 1

if __name__ == '__main__':
    stream_cam(0)
