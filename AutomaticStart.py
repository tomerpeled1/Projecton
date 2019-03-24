import SliceTypes as St
import ArduinoCommunication as Ac
import ImageProcessing as Ip
import time
import math


ANCHOR_POINT = 7.0, 4.0  # TODO write something informative here
ARM_LOC_0 = 7.5, 6.75  # manually put arm at this location
ARM_LOC_1 = Ac.DIMS[0]/2 - 0.5, math.sqrt(sum(Ac.ARMS)**2 - (Ac.DIMS[0]/2 - 0.5)**2) - Ac.d - 0.01  # location of arm
# before slicing apple to start game. the -0.5 is to make sure the pen stays in screen. the 0.01 is to avoid
# miscalculation in angles.
PASS_AD_POINT = Ac.DIMS[0]/2 + 1.0, ANCHOR_POINT[1]
DEBUG_POINT = Ac.DIMS[0]/2 - 5.0, ANCHOR_POINT[1]  # point for debugging
AD_TIME = 0.5  # time to wait before check for ad in secs


def automatic_start():
    """
    Executes the slices that start the game.
    :return: wanted value of perf_counter when you should take frame of ad
    """
    # move arm to location for slicing the apple
    Ac.make_slice_by_trajectory(St.slice_to_point(ARM_LOC_0, ARM_LOC_1), 0, False)
    time.sleep(1)
    # cut the apple for start
    arm_loc = Ac.start_cut(ARM_LOC_1)
    apple_time = time.perf_counter()
    # print("took apple_time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    Ac.make_slice_by_trajectory(St.slice_to_point(arm_loc, ANCHOR_POINT), 0, False)
    return apple_time + AD_TIME


def pass_ad(frame):
    """
    Checks for ad at beginning of the game, and gets rid of it if necessary.
    :param frame: frame taken from camera to check ad
    """
    # if Ip.check_ad(frame):
    #     print("passing ad")
    # else:
    #     print("no ad")
    time.sleep(1)
    Ac.make_slice_by_trajectory(St.slice_to_point(ANCHOR_POINT, PASS_AD_POINT), 0, False)
    time.sleep(1)
    Ac.make_slice_by_trajectory(St.slice_to_point(PASS_AD_POINT, ANCHOR_POINT), 0, False)
    time.sleep(1)
    Ac.make_slice_by_trajectory(St.slice_to_point(ANCHOR_POINT, PASS_AD_POINT), 0, False)
    time.sleep(1)
    Ac.make_slice_by_trajectory(St.slice_to_point(PASS_AD_POINT, ANCHOR_POINT), 0, False)


if __name__ == '__main__':
    # while True:
        # Ac.make_slice_by_trajectory(St.slice_to_point((-8, 6), (8, 6)), 0, False)
        # time.sleep(0.1)
        # Ac.make_slice_by_trajectory(St.slice_to_point((8, 6), (-8, 6)), 0, False)
        # time.sleep(0.1)
    # Ac.make_slice_by_trajectory(St.slice_to_point(ANCHOR_POINT, (0.0, 4.0)), 0, False)
    # Ac.make_slice_by_trajectory(St.slice_to_point((0.0, 4.0), ANCHOR_POINT), 0, False)
    x_arm = float(input("x location of arm: "))
    y_arm = float(input("y location of arm: "))

    while True:
        x_target = float(input("x location of target: "))
        y_target = float(input("y location of target: "))
        Ac.make_slice_by_trajectory(St.slice_to_point((x_arm, y_arm), (x_target, y_target)), 0, False)
        x_arm = x_target
        y_arm = y_target
        print("")
