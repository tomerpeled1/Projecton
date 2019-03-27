"""
MAIN!!!!!!!!!
"""

import Algorithmics as Algo
import ImageProcessing as Ip
from CameraInterface import Camera
import CameraInterface as Ci
import FruitDetection2 as Fd
import AutomaticStart as As
import time
import cv2
import State


SAVED_VIDEO_NAME = "EranFuckYou.avi "
LIVE = True
BACKGROUND_FILE_NAME = "bg.png"
CROP = True
FLIP = False
CALIBRATE = False
IMAGE_PROCESSING_ALGORITHMICS_INTEGRATION = True
ALGORITHMICS_MECHANICS_INTEGRATION = True
SIMULATION = True
CAPTURE_BACKGROUND = True
RESIZE = False
AUTOMATIC_START = False
MULTI = True
RESTARTED = False
CAMERA = None
RAN = False



CHOSEN_SLICE = Algo.LINEAR

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
    global CAMERA
    if integration[0]:
        Ip.init_everything(integrate_with_algorithmics=integration[0], multi=MULTI)
        Algo.init_everything(slice_type=CHOSEN_SLICE, integrate_with_mechanics=integration[1],
                             simulate=simulation, multi=MULTI)

    fruits_info = []  # Initialize fruits known.
    # Create new camera object.
    if not RESTARTED:
        CAMERA = Camera(src, flip=image_processing_features[0], crop=image_processing_features[1],
                        live=image_processing_features[2], calibrate=image_processing_features[3],
                        resize=image_processing_features[4], multi=MULTI)
        if CAMERA.LIVE:
            CAMERA.set_camera_settings(settings)

    if AUTOMATIC_START and not MULTI:  # Execute automatic start
        ad_time = As.automatic_start()
        time_to_wait_for_ad = ad_time - time.perf_counter()
        if CAMERA.is_opened():
            if time_to_wait_for_ad > 0:
                print("waiting for ad:", time_to_wait_for_ad)
                time.sleep(time_to_wait_for_ad)
            As.pass_ad((CAMERA.read())[1])

    # return  # to test automatic start

    # Get background
    bg = cv2.imread(BACKGROUND_FILE_NAME)
    if CAPTURE_BACKGROUND:
        # Allows user to click in order to capture background.
        print("choose background")
        bg = CAMERA.background_and_wait()
        cv2.imwrite(BACKGROUND_FILE_NAME, bg)
    else:
        cv2.imshow("Saved Background", bg)
        print("press any key to continue.")
        cv2.waitKey(0)

    current = bg
    counter = 0
    buffer = []  # Buffer of images for debugging purposes.
    current_state = State.State()
    time_of_frame = time.perf_counter()
    counter = 1 ## TODO remove
    # Main while loop.
    while CAMERA.is_opened() and counter < 1000000:
        t = time.perf_counter()
          # dont image proccess during a slice in multiplayer mode
            # t1 = time.perf_counter()
            # print("********************************************************************")
        counter += 1
        current = CAMERA.next_frame(current, time_of_frame)  # Retrieve next frame.
        time_of_frame = time.perf_counter()
        temp_frame = current.copy()  # Copy the frame.
        if not (Algo.during_slice and MULTI):
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
            current_state.update_state(Ip.FRUIT_TO_EXTRACT, time_of_frame)
            Ip.clear_fruits()
            if not Algo.during_slice:
                slice_flag, points_to_slice, sliced_fruits = current_state.is_good_to_slice()
                if slice_flag:
                    if integration[0]:
                        add_slice_to_queue(points_to_slice, sliced_fruits)
                        current_state.remove_sliced_fruits(sliced_fruits)

            cv2.imshow("temp_frame", temp_frame)
            buffer.append(temp_frame)  # Inserts frame to buffer.
            # t2 = time.perf_counter()
            if Ip.fruits_for_debug_trajectories:
                # for i in range(1 ,min(len(Ip.fruits_for_debug_trajectories), 3)):
                Ip.draw_trajectory(Ip.fruits_for_debug_trajectories[-1], CAMERA.last_big_frame, time_of_frame)
            cv2.imshow("please work", CAMERA.last_big_frame)
            # print("time for everything", abs(t1 - t2))
            if cv2.waitKey(1) == 27:
                restart()
        # print(str(counter) + " time for detection: " + str(time.perf_counter() - t))
            counter += 1
    # Ip.debug_with_buffer(buffer)
    # Ip.show_original(CAMERA, buffer)


def add_slice_to_queue(slice_points_to_add, sliced_fruits):
    """
    Adds the given slice to the slice queue.
    :param slice_points_to_add: slice to add to queue in list of points
    :param sliced_fruits: fruits the slice should cut (for simulation)
    """
    Algo.add_slice_to_queue(slice_points_to_add, sliced_fruits)


def restart():
    global RESTARTED
    global RAN
    if RAN:
        RESTARTED = True
        run()


def run():
    global RAN
    if not RAN:
        RAN = not RAN
    if LIVE:
        fruit_shaninja(0, Ci.DARK_101_SETTINGS_new2)
        print("finished")
    else:
        fruit_shaninja(SAVED_VIDEO_NAME, Ci.DARK_101_SETTINGS_new2)



if __name__ == '__main__':
    run()
