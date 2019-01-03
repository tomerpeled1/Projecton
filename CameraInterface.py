import cv2
from imutils.video import WebcamVideoStream
import time
from Calibrate import calibrate
import SavedVideoWrapper
import Algorithmics as sc

# Settings for camera in projecton lab when lights on.
LIGHT_LAB_SETTINGS = (215, 75, -7, 10)  # order is (saturation, gain, exposure, focus)
# Settings for camera in projecton lab with shadow from table over screen.
TABLE_ABOVE_SETTINGS = (255, 100, -6, 10)  # order is (saturation, gain, exposure, focus)
MORNING_101_SETTINGS = (220, 40, -7, 5)  # order is (saturation, gain, exposure, focus)
DARK_101_SETTINGS = (255, 144, -8, 16)  # order is (saturation, gain, exposure, focus)
DARK_101_SETTINGS_BEESITO = (255, 127, -7, 5)
MORNING_101_SETTINGS_BEESITO = (255, 127, -7, 10)
IPAD_NIGHT_LIT = (255, 24, -7, 10)
IPAD_NIGHT_LIT_SILVER = (255, 37, -7, 10)
IPAD_NIGHT_DARK = (255, 8, -6, 10)
IPAD_B4_MIDDLE_LIGHTS_OFF_CLOSED_DRAPES = (255, 7, -6, 5)
IPAD_B4_MIDDLE_LIGHTS_OFF_CLOSED_DRAPES_2 = (255, 18, -6, 10)

# Parameter whether or not set specific white balance - default is 2000.
WHITE_BALANCE = True





class Camera:

    def __init__(self, src=0, flip = True, crop = False, live = True, calibrate = False):
        """
        Constructor for camera object.
        :param src: The source for the camera. 0 for live and video name for saved video.
        :param FLIP: True if image needs to be flipped.
        :param CROP: True if image needs to be cropped.
        :param LIVE: True if live camera is on.
        :param CALIBRATE: Feature to calibrate the tablet. True if we want to use the feature.
        """
        self.src = src
        self.FLIP = flip
        self.CROP = crop
        self.LIVE = live
        self.CALIBRATE = calibrate
        # Opens a stream for the camera.
        if self.LIVE:
            self.stream = WebcamVideoStream(src=src, name="Live Video").start()
        else:
            self.stream = SavedVideoWrapper.SavedVideoWrapper(src)
        # Crop dimensions for automatic calibration.
        self.bl_crop_dimensions = []
        self.tr_crop_dimensions = []
        # Current frame taken.
        self.current = None
        # Buffer which saves the original frames to display for debug purposes.
        self.buffer = []
        # Maximal size for buffer to avoid using too much memory.
        self.MAX_SIZE_BUFFER = 500

    def read(self):
        """
        Reads and edits a new frame from the stream.
        :return: the frame after being read.
        """
        frame = self.stream.read()
        # Option for calibration.
        if self.CALIBRATE:
            frame = self.crop_to_screen_size(frame)
        self.current = frame
        # Option for crop.
        if self.CROP:
            frame = self.crop_image(frame)
        # Option for flip.
        if self.FLIP:
            frame = Camera.flip(frame)
        return frame

    @staticmethod
    def flip(frame):
        """
        Flips a given frame upside down.
        :param frame: the frame given.
        :return: the frame after being flipped (saved in the same memory location).
        """
        cv2.flip(frame, -1, frame)
        return frame

    def is_opened(self):
        """
        Checks if stream is opened.
        """
        return self.stream.stream.isOpened()

    def next_frame(self, current):
        """
        Returns the frame which comes after the current one (when threading we get the same frame multiple times).
        :param current: The current frame.
        :return: The first frame from the stream which is different than current.
        """
        while True:
            to_return = self.read()
            dif = cv2.subtract(to_return, current)
            dif = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
            # Checks if there is a difference between current and to_return.
            if cv2.countNonZero(dif) > 0:
                # Saves image to buffer.
                if len(self.buffer) < self.MAX_SIZE_BUFFER:
                    self.buffer.append(self.current)
                return to_return

    def next_frame_for_bg(self, current):
        """
        Takes out next frame for background capture (not saved to buffer).
        :param current: Current frame.
        :return: next frame different from current.
        """
        while True:
            to_return = self.read()
            dif = cv2.subtract(to_return, current)
            dif = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
            if cv2.countNonZero(dif) > 0:
                return to_return

    def crop_to_screen_size(self, frame):
        """
        Crops image with respect to screen size for calibration.
        :param frame: The frame to crop.
        :return: The frame cropped with the calibration dimensions.
        """
        frame = frame[self.tr_crop_dimensions[1]:self.bl_crop_dimensions[1],
                self.bl_crop_dimensions[0]:self.tr_crop_dimensions[0]]
        # Updates the screen size in algorithm module.
        sc.init_info(frame.shape[:2])
        return frame

    def crop_image(self, frame):
        """
        Crops an image to a bottom third.
        :param frame: The frame to crop.
        :return: The bottom third of the frame (saved in the same memory location).
        """
        (height, width, depth) = frame.shape
        if self.FLIP:
            frame = frame[:160, width//2 - 240 : width//2 + 240]
        else:
            frame = frame[height - 160 : height, width//2 - 240 : width//2 + 240]
        return frame

    def background_and_wait(self):
        """
        Waits to capture background when space is clicked.
        :return: The captured background.
        """
        return self.wait_for_click()

    def set(self, settings, white_balance = False):
        """
        Sets camera settings.
        :param settings: The settings to set.
        :param white_balance: Boolean parameter which determines whether or not to set white balance.
        """
        cam = self.stream.stream
        # cam.set(3, 1920)  # width
        # cam.set(4, 1080)  # height
        cam.set(12, settings[0])  # saturation     min: 0   , max: 255 , increment:1
        cam.set(14, settings[1])  # gain           min: 0   , max: 127 , increment:1
        cam.set(15, settings[2])  # exposure       min: -7  , max: -1  , increment:1
        cam.set(28, settings[3])  # focus
        if white_balance:
            cam.set(17, 2000)  # white_balance  min: 4000, max: 7000, increment:1

    def set_camera_settings(self, settings):
        """
        Sets the camera settings with respect to settings given in an array.
        :param settings: array of size 4 which represents (saturation, gain, exposure, focus).
        """
        self.set(settings, WHITE_BALANCE)
        if self.CALIBRATE:
            frame = None
            while True:
                frame = self.stream.read()
                cv2.imshow("calibrate", frame)
                # Calibrate the camera with a click on space.
                if cv2.waitKey(1) == 32:
                    (bl, tr) = calibrate(frame)
                    self.bl_crop_dimensions = bl
                    self.tr_crop_dimensions = tr
                    return

    def wait(self, x):
        """
        Waits x seconds.
        """
        cur = self.read()
        counter = 0
        while counter < 30 * x:
            cur = self.next_frame_for_bg(cur)
            counter += 1
        return cur

    def wait_for_click(self):
        """
        Waits for click to capture image.
        :return: image captured.
        """
        frame = self.read()
        counter = 0
        while True:
            frame = self.next_frame_for_bg(frame)
            cv2.imshow("until background", frame)
            counter += 1
            # Captures and returns image when space is pressed.
            if cv2.waitKey(1) == 32:
                cv2.imshow("until background", frame)
                cv2.waitKey(0)
                return frame

if __name__ == '__main__':
    pass
