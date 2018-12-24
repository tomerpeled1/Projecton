import cv2
from imutils.video import WebcamVideoStream
import time

class Camera:

    def __init__(self, src=0):
        self.src = src
        self.stream = WebcamVideoStream(src).start()

    def is_opened(self):
        return True #TODO change function.

    def next_frame(self, current):
        while True:
            to_return = self.stream.read()
            to_return = self.crop_image(to_return)
            dif = cv2.subtract(to_return, current)
            dif = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
            if (cv2.countNonZero(dif) > 0):
                return to_return

    def next_frame_for_bg(self, current):
        while True:
            to_return = self.stream.read()
            dif = cv2.subtract(to_return, current)
            dif = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
            if (cv2.countNonZero(dif) > 0):
                return to_return

    @staticmethod
    def crop_image(frame):
        # (height, width, depth) = frame.shape
        # new_h = int(height/3)
        # new_w = int(width/7)
        # frame = frame[2*new_h:height, new_w:6*new_w]
        #frame = cv2.resize(frame,(0,0),fx = 0.5, fy = 0.5)
        return frame


    def get_frame(self):
        frame = self.stream.read()
        frame = Camera.crop_image(frame)
        return frame

    def background_and_wait(self):
        return self.wait_for_click()

    def stream_cam(self):
        ##set_camera_settings(stream.stream)
        counter = 0
        current = self.stream.read()
        frames = [current]

        while counter < 1000:
            t1 = time.perf_counter()
            next = self.next_frame(current)
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
        self.stream.stop()


    def set_camera_settings(self):
        pass
        # #       key value
        # # cam.set(3, 1920)  # width
        # # cam.set(4, 1080)  # height
        # cam.set(10, 120)  # brightness     min: 0   , max: 255 , increment:1
        # cam.set(11, 120)  # contrast       min: 0   , max: 255 , increment:1
        # cam.set(12, 120)  # saturation     min: 0   , max: 255 , increment:1
        # cam.set(13, 13)  # hue
        # cam.set(14, 50)  # gain           min: 0   , max: 127 , increment:1
        # cam.set(15, -6)  # exposure       min: -7  , max: -1  , increment:1
        # cam.set(17, 5000)  # white_balance  min: 4000, max: 7000, increment:1
        # cam.set(28, 0)  # focus          min: 0   , max: 255 , increment:5


    def wait(self, x):
        cur = self.stream.read()
        counter = 0
        while counter < 30 * x:
            cur = self.next_frame_for_bg(cur)
            counter += 1
        return cur

    def wait_for_click(self):
        cur = self.stream.read()
        counter = 0
        while True:
            cur = self.next_frame_for_bg(cur)
            cv2.imshow("until background", cur)
            counter += 1
            if cv2.waitKey(1) == 32:
                cv2.imshow("until background", cur)
                cv2.waitKey(0)
                return cur

if __name__ == '__main__':
    camera = Camera(0)
    camera.stream_cam()
