import SliceTypes as St
import Algorithmics as Al
import ArduinoCommunication as Ac
import ImageProcessing as Ip

ANCHOR_POINT = 0, 0  # TODO write something imformative here


def automatic_start():
    # cut the apple for start
    arm_loc = Ac.start_cut()

    Al.do_slice(St.slice_to_point(arm_loc, ANCHOR_POINT))  # TODO write slice_to_point()
    if Ip.check_ad():
        Al.do_slice(St.ad_slice())  # TODO write ad_slice()


