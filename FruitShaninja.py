"""
MAIN!!!!!!!!!
"""

import Algorithmics as Algo
import ImageProcessing as Ip
from CameraInterface import Camera
import CameraInterface as Ci
import FruitDetection2 as Fd
import AutomaticStart as As
import ArduinoCommunication as Ac
import time
import cv2


SAVED_VIDEO_NAME = "2019-03-17 19-59-34.flv "
LIVE = True
BACKGROUND_FILE_NAME = "bg.png"
CROP = True
FLIP = True
CALIBRATE = False
IMAGE_PROCESSING_ALGORITHMICS_INTEGRATION = True
ALGORITHMICS_MECHANICS_INTEGRATION = True
SIMULATION = False
BACKGROUND = True
RESIZE = False

IMAGE_PROCESSING_FEATURES = (FLIP, CROP, LIVE, CALIBRATE, RESIZE)
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
                    live=image_processing_features[2], calibrate=image_processing_features[3],
                    resize=image_processing_features[4])
    if camera.LIVE:
        camera.set_camera_settings(settings)

    As.automatic_start()
    # Ac.wait(1000)
    current = (camera.read())[1] # Retrieve next frame.
    As.pass_ad(current)

    bg = cv2.imread(BACKGROUND_FILE_NAME)
    if BACKGROUND:
        # Allows user to click in order to capture background.
        print("choose background")
        bg = camera.background_and_wait()
        cv2.imwrite(BACKGROUND_FILE_NAME, bg)
    else:
        cv2.imshow("Saved Background", bg)
        print("press any key to continue.")
        cv2.waitKey(0)

    current = bg

    counter = 0
    buffer = []  # Buffer of images for debugging purposes.

    # Main while loop.
    while camera.is_opened() and counter < 1200:
        t1 = time.perf_counter()
        # print("********************************************************************")
        counter += 1
        current = camera.next_frame(current) # Retrieve next frame.
        time_of_frame = time.perf_counter()
        temp_frame = current.copy()  # Copy the frame.
        detection_results = Fd.fruit_detection2(temp_frame, bg, Ip.CONTOUR_AREA_THRESH,
                                                time_of_frame)  # Run detection on fruits.
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
        if (Ip.fruits_for_debug_trajectories):
            # for i in range(1 ,min(len(Ip.fruits_for_debug_trajectories), 3)):
            Ip.draw_trajectory(Ip.fruits_for_debug_trajectories[-1], camera.last_big_frame)
        cv2.imshow("please work", camera.last_big_frame)
        # print("time for everything", abs(t1 - t2))
        if cv2.waitKey(1) == 27:
            break
    # Ip.debug_with_buffer(buffer)
    Ip.show_original(camera)


if __name__ == '__main__':
    if LIVE:
        fruit_shaninja(0, Ci.DARK_101_SETTINGS_new2)
    else:
        fruit_shaninja(SAVED_VIDEO_NAME, Ci.DARK_101_SETTINGS_new2)
