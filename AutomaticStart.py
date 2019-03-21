import SliceTypes as St
import Algorithmics as Al
import ArduinoCommunication as Ac

ANCOR_POINT = 0, 0  # TODO write something imformative here


def automatic_start():
    # cut the apple for start
    arm_loc = Ac.start_cut()

    Al.do_slice(St.slice_to_point(arm_loc, ANCOR_POINT))  # TODO write slice_to_point()

