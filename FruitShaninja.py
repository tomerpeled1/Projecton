import Algorithmics as Algo
import ImageProcessing as Ip
from CameraInterface import Camera
import CameraInterface as Ci
import FruitDetection as Fd
import time
import cv2

"""
MAIN!!!!!!!!!
"""

SAVED_VIDEO_NAME = "sundayNoon.flv"
LIVE = False
CROP = False
FLIP = True
CALIBRATE = False
IMAGE_PROCESSING_ALGORITHMICS_INTEGRATION = True
ALGORITHMICS_MECHANICS_INTEGRATION = True
SIMULATION = True

IMAGE_PROCESSING_FEATURES = [FLIP, CROP, LIVE, CALIBRATE]
INTEGRATION = [IMAGE_PROCESSING_ALGORITHMICS_INTEGRATION, ALGORITHMICS_MECHANICS_INTEGRATION]

def fruit_shaninja(src, settings, image_processing_features = IMAGE_PROCESSING_FEATURES,
                  integration = INTEGRATION, simulation = SIMULATION):
    """
    Main function which runs.
    :param src: The source of the video (0 for live, video name for saved video).
    :param settings: The camera settings we want to set when live.
    :param live: True if we use live camera.
    :param crop: True if we want to crop image.
    :param flip: True if image needs to be flipped.
    :param calibrate: True if we want to use automatic calibration.
    """
    # Initiate algorithmics if integrated.
    if integration[0]:
        Ip.init_everything(integrate_with_algorithmics= integration[0])
        Algo.init_everything(integrate_with_mechanics = integration[1], simulate = simulation)
    # Initialize fruits known.
    fruits_info = []
    # Creates new camera object.
    camera = Camera(src, flip = image_processing_features[0], crop = image_processing_features[1],
                    live = image_processing_features[2], calibrate = image_processing_features[3])
    if camera.LIVE:
        camera.set_camera_settings(settings)
    # Allows user to click in oreder to capture background.
    print("choose background")
    bg = camera.background_and_wait()
    current = bg
    counter = 0
    # Buffer of images for debugging purposes.
    buffer = []
    # Main while loop.
    while camera.is_opened() and counter < 90000:
        t1 = time.perf_counter()
        counter += 1
        # Retrieves next frame.
        current = camera.next_frame(current)
        # Copies the frame.
        temp_frame = current.copy()
        # Runs detection on fruits.
        detection_results = Fd.fruit_detection(temp_frame, bg, Ip.CONTOUR_AREA_THRESH)
        # Draws the fruits as detected (mainly for debugging).
        cv2.drawContours(temp_frame, detection_results.conts, -1, (0, 255, 0), 2)
        # Tracks known fruits in the current frame and removes them for fruits_info.
        Ip.track_known_fruits(fruits_info, temp_frame, detection_results)
        if len(detection_results.conts) > 0:
            # In case there are more fruits left it inserts them as new fruits.
            Ip.insert_new_fruits(detection_results, fruits_info, temp_frame)
        for fruit in fruits_info:
            # Draws all fruits which are not falling.
            if not fruit.is_falling:
                Ip.draw(fruit, temp_frame)
        cv2.imshow("temp_frame", temp_frame)
        # Inserts frame to buffer.
        buffer.append(temp_frame)
        t2 = time.perf_counter()
        print("time for everything", abs(t1 - t2))
        if cv2.waitKey(1) == 27:
            break
    Ip.debug_with_buffer(buffer)
    # Ip.show_original(camera)

if __name__ == '__main__':
    fruit_shaninja(SAVED_VIDEO_NAME, Ci.IPAD_B4_MIDDLE_LIGHTS_OFF_CLOSED_DRAPES)
