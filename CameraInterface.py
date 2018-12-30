import cv2
from imutils.video import WebcamVideoStream
import time
from Calibrate import calibrate
import SavedVideoWrapper

LIGHT_LAB_SETTINGS = (215, 75, -7, 12)  # order is (saturation, gain, exposure, focus)
TABLE_ABOVE_SETTINGS = (255, 100, -6, 12)  # order is (saturation, gain, exposure, focus)
MORNING_101_SETTINGS = (220, 40, -7, 5)  # order is (saturation, gain, exposure, focus)
DARK_101_SETTINGS = (255, 144, -8, 16)  # order is (saturation, gain, exposure, focus)
DARK_101_SETTINGS_BEESITO = (255, 127, -7, 5)
MORNING_101_SETTINGS_BEESITO = (255, 127, -7, 12)
IPAD_NIGHT_LIT = (255, 24, -7, 12)
IPAD_NIGHT_LIT_SILVER = (255, 37, -7, 12)

CALIBRATE = False


class Camera:
    def __init__(self, src=0, FLIP=True, CROP=False, LIVE=True):
        self.src = src
        self.FLIP = FLIP
        self.CROP = CROP
        self.LIVE = LIVE
        if self.LIVE:
            self.stream = WebcamVideoStream(src=src, name="Live Video").start()
        else:
            self.stream = SavedVideoWrapper.SavedVideoWrapper(src)
        self.bl_crop_dimensions = []
        self.tr_crop_dimensions = []
        self.current = None
        self.buffer = []

    def read(self):
        frame = self.stream.read()
        # frame = cv2.resize(frame, (640, 480))
        if CALIBRATE:
            frame = self.crop_to_screen_size(frame)
        self.current = frame.copy()
        if (self.CROP):
            frame = self.crop_image(frame)
        if (self.FLIP):
            frame = Camera.flip(frame)
            # prev = Camera.flip(flipped)
            # # frame = Camera.flip(frame)
            # dif = cv2.subtract(frame, prev)
            # dif = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
            # a = cv2.countNonZero(dif)
        return frame

    @staticmethod
    def flip(frame):
        # t1 = time.perf_counter()
        # rows, cols , _= frame.shape
        # # cols-1 and rows-1 are the coordinate limits.
        # M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 180, 1)
        # dst = cv2.warpAffine(frame, M, (cols, rows))
        cv2.flip(frame, -1, frame)
        # t2 = time.perf_counter()
        # print("flip time:", abs(t2-t1))
        return frame
        # return frame

    def is_opened(self):
        return self.stream.stream.isOpened()

    def next_frame(self, current):
        while True:
            to_return = self.read()
            # to_return = self.crop_image(to_return)
            # if (not current is to_return):
            #     return to_return
            dif = cv2.subtract(to_return, current)
            dif = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
            if (cv2.countNonZero(dif) > 0):
                self.buffer.append(self.current)
                return to_return

    def next_frame_for_bg(self, current):
        while True:
            to_return = self.read()
            # if not to_return is current:
            #     return to_return
            dif = cv2.subtract(to_return, current)
            dif = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
            if (cv2.countNonZero(dif) > 0):
                return to_return

    def crop_to_screen_size(self, frame):
        frame = frame[self.tr_crop_dimensions[1]:self.bl_crop_dimensions[1],
                self.bl_crop_dimensions[0]:self.tr_crop_dimensions[0]]
        return frame

    def crop_image(self, frame):
        (height, width, depth) = frame.shape
        new_h = int(height / 3)
        new_w = int(width / 8)
        if self.FLIP:
            # this is exactly for beesito
            # frame = frame[int(height / 5)-10:int(new_h + height / 5)-10, new_w:7 * new_w]

            # and for generic video:
            # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            frame = frame[:new_h, new_w:7 * new_w]
        elif not self.FLIP:
            frame = frame[2 * new_h:height, new_w: 7 * new_w]
        return frame

    def background_and_wait(self):
        return self.wait_for_click()

    def stream_cam(self):
        # set_camera_settings(stream.stream)
        counter = 0
        current = self.read()
        frames = [current]
        while counter < 1000:
            t1 = time.perf_counter()
            next = self.next_frame_for_bg(current)
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

    def set(self, settings):
        cam = self.stream.stream
        # cam.set(3, 1920)  # width
        # cam.set(4, 1080)  # height
        # cam.set(10, 128)  # brightness     min: 0   , max: 255 , increment:1
        # cam.set(11, 128)  # contrast       min: 0   , max: 255 , increment:1
        cam.set(12, settings[0])  # saturation     min: 0   , max: 255 , increment:1
        cam.set(14, settings[1])  # gain           min: 0   , max: 127 , increment:1
        cam.set(15, settings[2])  # exposure       min: -7  , max: -1  , increment:1
        # cam.set(17, 4000)  # white_balance  min: 4000, max: 7000, increment:1
        cam.set(28, settings[3])

    def set_camera_settings(self, settings):
        self.set(settings)
        if CALIBRATE:
            frame = None
            while True:
                frame = self.stream.read()
                cv2.imshow("calibrate", frame)
                if cv2.waitKey(1) == 32:
                    (bl, tr) = calibrate(frame)
                    self.bl_crop_dimensions = bl
                    self.tr_crop_dimensions = tr
                    return
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
        cur = self.read()
        counter = 0
        while counter < 30 * x:
            cur = self.next_frame_for_bg(cur)
            counter += 1
        return cur

    def wait_for_click(self):
        cur = self.read()
        counter = 0
        while True:
            cur = self.next_frame_for_bg(cur)
            cv2.imshow("until background", cur)
            counter += 1
            if cv2.waitKey(1) == 32:
                cv2.imshow("until background", cur)
                cv2.waitKey(0)
                return cur

    def calibrate_camera(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 200, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("gray", thresh)
        cv2.waitKey(0)
        white = []
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > 0:
                white.append(cont)
        for w in white:
            print(99999999999)
            print(w)
        return white


if __name__ == '__main__':
    pass
    # camera = Camera(0)
    # # camera.stream_cam()
    # frame = camera.read()
    # cv2.imshow("frame", frame)
    # cv2.waitKey(0)
    # white = camera.calibrate_camera(frame)
    # cv2.drawContours(frame,white,0,(0,255,0))
    # cv2.imshow("cool thing", frame)
    # cv2.waitKey(0)
