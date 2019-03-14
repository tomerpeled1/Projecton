"""
MAIN!!!!!!!!!
"""

import Algorithmics as Algo
import ImageProcessing as Ip
from CameraInterface import Camera
import CameraInterface as Ci
import FruitDetection as Fd
import time
import cv2


SAVED_VIDEO_NAME = "sundayNoon.flv"
LIVE = True
CROP = True
FLIP = True
CALIBRATE = False
IMAGE_PROCESSING_ALGORITHMICS_INTEGRATION = True
ALGORITHMICS_MECHANICS_INTEGRATION = True
SIMULATION = False

IMAGE_PROCESSING_FEATURES = (FLIP, CROP, LIVE, CALIBRATE)
INTEGRATION = (IMAGE_PROCESSING_ALGORITHMICS_INTEGRATION, ALGORITHMICS_MECHANICS_INTEGRATION)


def fruit_shaninja(src, settings, image_processing_features=IMAGE_PROCESSING_FEATURES,
                   integration=INTEGRATION, simulation=SIMULATION):
    """
    Main function which runs.
    :param src: The source of the video (0 for live, video name for saved video).
    :param settings: The camera settings we want to set when live.
    :param image_processing_features: (flip, crop, live, calibrate) tuple of booleans that decides:
    live: True if we use live camera.
    crop: True if we want to crop image.
    flip: True if image needs to be flipped.
    calibrate: True if we want to use automatic calibration.
    :param integration: (image_processing_algorithmics, algorithmics_mechanics) tuple of booleans that decides:
    image_processing_algorithmics: integrate image processing with algorithmics
    algorithmics_mechanics: integrate algorithmics with mechanics
    :param simulation: boolean that decides weather to run the simulation or not.
    """
    # Initiate algorithmics if integrated.
    if integration[0]:
        Ip.init_everything(integrate_with_algorithmics=integration[0])
        Algo.init_everything(integrate_with_mechanics=integration[1], simulate=simulation)
    fruits_info = []  # Initialize fruits known.
    # Create new camera object.
    camera = Camera(src, flip=image_processing_features[0], crop=image_processing_features[1],
                    live=image_processing_features[2], calibrate=image_processing_features[3])
    if camera.LIVE:
        camera.set_camera_settings(settings)

    # Allows user to click in order to capture background.
    print("choose background")
    bg = camera.background_and_wait()
    current = bg

    counter = 0
    buffer = []  # Buffer of images for debugging purposes.

    # Main while loop.
    while camera.is_opened() and counter < 90000:
        t1 = time.perf_counter()
        counter += 1
        current = camera.next_frame(current)  # Retrieve next frame.
        temp_frame = current.copy()  # Copy the frame.
        detection_results = Fd.fruit_detection(temp_frame, bg, Ip.CONTOUR_AREA_THRESH)  # Run detection on fruits.
        cv2.drawContours(temp_frame, detection_results.conts, -1, (0, 255, 0), 2)  # Draw the fruits as detected.
        Ip.track_known_fruits(fruits_info, temp_frame, detection_results)  # Track known fruits in current frame and
        # remove them for fruits_info.
        if len(detection_results.conts) > 0:  # In case there are more fruits left it inserts them as new fruits.
            Ip.insert_new_fruits(detection_results, fruits_info, temp_frame)
        for fruit in fruits_info:  # Draws all fruits which are not falling.
            if not fruit.is_falling:
                Ip.draw(fruit, temp_frame)
        cv2.imshow("temp_frame", temp_frame)
        buffer.append(temp_frame)  # Inserts frame to buffer.
        t2 = time.perf_counter()
        print("time for everything", abs(t1 - t2))
        if cv2.waitKey(1) == 27:
            break
    Ip.debug_with_buffer(buffer)
    # Ip.show_original(camera)


if __name__ == '__main__':
    fruit_shaninja(0, Ci.IPAD_B4_MIDDLE_LIGHTS_OFF_CLOSED_DRAPES)
