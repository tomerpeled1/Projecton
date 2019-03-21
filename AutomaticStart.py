import SliceTypes as St
import Algorithmics as Al
import ArduinoCommunication as Ac
import ImageProcessing as Ip
import time
import cv2


ANCHOR_POINT = 0, 0  # TODO write something imformative here


def automatic_start():
    # cut the apple for start
    arm_loc = Ac.start_cut()

    # Al.do_slice(St.slice_to_point(arm_loc, ANCHOR_POINT))  # TODO write slice_to_point()


def pass_ad(frame):
    if Ip.check_ad(frame):
        Al.do_slice(St.ad_slice())  # TODO write ad_slice()


if __name__ == '__main__':
    pass_ad(cv2.imread("AD_IMAGE.png"))
