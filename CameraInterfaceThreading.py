import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import time

def stream_cam(src):
    stream = WebcamVideoStream(src).start()
    fps = FPS().start()
    ##set_camera_settings(stream.stream)
    while True:
        t1 = time.perf_counter()
        frame = stream.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
           break  # esc to quit
        fps.update()
        t2 = time.perf_counter()
        print("time for frame: " + str(t2 - t1))
    # fps.stop()
    cv2.destroyAllWindows()
    stream.stop()

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

if __name__ == '__main__':
    stream_cam(0)
