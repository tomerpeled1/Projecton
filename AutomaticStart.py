import SliceTypes as St
import Algorithmics as Al
import ArduinoCommunication as Ac
import ImageProcessing as Ip
import time
import cv2


ANCHOR_POINT = 7.5, 6.5  # TODO write something informative here
ARM_LOC_0 = 7.5, 6.5  # manually put arm at this location
ARM_LOC_1 = 8, 8.6  # location of arm before slicing apple to start game
PASS_AD_POINT = -1, 5


def automatic_start():
    # move arm to location for slicing the apple
    Ac.make_slice_by_trajectory(St.slice_to_point(ARM_LOC_0, ARM_LOC_1), 0, False)
    time.sleep(1)
    # cut the apple for start
    arm_loc = Ac.start_cut(ARM_LOC_1)

    Ac.make_slice_by_trajectory(St.slice_to_point(arm_loc, ANCHOR_POINT), 0, False)


def pass_ad(frame):
    if Ip.check_ad(frame):
        print("passed ad")
        # Ac.make_slice_by_trajectory(St.slice_to_point(PASS_AD_POINT), 0)


if __name__ == '__main__':
    # Ac.make_slice_by_trajectory(St.slice_to_point(ARM_LOC_0, (8, 8.6)), 0, False)
    automatic_start()
