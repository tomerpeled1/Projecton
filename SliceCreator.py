import math
import SliceTypes
import time
from scipy.optimize import curve_fit
import ArduinoCommunication
import simulation_like_motor_commands as slm

RELATIVE_ACC = 2.34
ARM_DELAY = 1
CROP_SIZE = (160, 480)
FRAME_SIZE = (480, 640)
SCREEN_SIZE = (16, 12)
ACC = RELATIVE_ACC * SCREEN_SIZE[1]

# CHOSEN_SLICE_TYPE = SliceTypes.stupid_slice

on_screen_fruits = []
SIMULATE = True


class Trajectory:
    def __init__(self, x0, v, theta):
        self.x0 = x0
        self.v = v
        self.theta = theta

    def calc_trajectory(self):
        return lambda t: (x_trajectory(t, self.x0, self.v, self.theta), y_trajectory(t, self.v, self.theta))

    def calc_peak(self):
        t = self.v * math.sin(self.theta) / ACC
        return t,  self.calc_trajectory()(t)

    def calc_life_time(self):
        t = 2 * self.v * math.sin(self.theta) / ACC
        return t


def update_fruits(fruits):
    fruits_locs = [[pixel2cm(pix_loc) for pix_loc in fruit.centers] for fruit in fruits]
    # fruit_trajectories = [get_trajectory(fruit_locs) for fruit_locs in fruits_locs]
    # on_screen_fruits.extend([[fruit_trajectories[i], fruits[i].time_created] for i in range(len(fruits))])


def create_slice():
    slice = calc_slice(on_screen_fruits)
    return slice


def do_slice(slice):
    parametization, timer, t_peak = slice
    # time.sleep(time_until_slice(slice))
    if SIMULATE:
        slm.run_simulation(parametization)
    else:
        ArduinoCommunication.make_slice_by_trajectory(parametization)


def create_and_do_slice():
    slice = create_slice()
    do_slice(slice)


def pixel2cm(pix_loc):
    (i_coord_crop, j_coord_crop) = pix_loc
    i_coord_frame = i_coord_crop + FRAME_SIZE[0] - CROP_SIZE[0]
    j_coord_frame = FRAME_SIZE[1] / 2 - CROP_SIZE[1] / 2 + j_coord_crop
    i_coord_screen =  (i_coord_frame / FRAME_SIZE[0]) * SCREEN_SIZE[0]
    j_coord_screen =  j_coord_frame / FRAME_SIZE[1] * SCREEN_SIZE[1]
    return j_coord_screen, i_coord_screen


def get_trajectory(fruit_locs):
    x_coords = [fruit_loc[0] for fruit_loc in fruit_locs]
    y_coords = [fruit_loc[1] for fruit_loc in fruit_locs]
    popt, pcov = curve_fit(trajectory_physics, x_coords, y_coords)
    x0_par, v_par, theta_par = popt
    trajectory = Trajectory(x0_par, v_par, theta_par)
    return trajectory


def trajectory_physics(x, x0, v, theta):
    return (x - x0) * math.tan(theta) - ACC * (x - x0) ** 2 / (2 * v ** 2 * math.cos(theta) ** 2)


def x_trajectory(t, x0, v, theta):
    return x0 + v * math.cos(theta) * t


def y_trajectory(t, v, theta):
    return v * math.sin(theta) * t - 0.5 * ACC * t ** 2


def calc_slice(fruit_trajectories_and_starting_times):
    # time.sleep(time_until_slice())
    return SliceTypes.stupid_slice(get_arm_loc(), fruit_trajectories_and_starting_times)


def get_arm_loc():
    return -5, 5


def time_until_slice(fruit):
    _, timer, t_peak = fruit
    return timer + t_peak - time.clock()


def init_info(crop_size, frame_size, screen_size):
    global CROP_SIZE, FRAME_SIZE, SCREEN_SIZE
    CROP_SIZE = crop_size
    FRAME_SIZE = frame_size
    SCREEN_SIZE = screen_size


def remove_sliced_fruits(fruits):
    for fruit in fruits:
        on_screen_fruits.remove(fruit)
    for fruit in on_screen_fruits:
        traj, timer = fruit
        if (time.clock() > timer + traj.calc_life_time()):
            on_screen_fruits.remove(fruit)


if __name__ == "__main__":
    for _ in range(10):
        create_and_do_slice()
        time.sleep(1)
