import math
import matplotlib.pyplot as plt
import statistics as st
# import cv2


import SliceTypes
import time
from scipy.optimize import curve_fit
import ArduinoCommunication
import simulation_like_motor_commands as slm
from threading import Thread
from multiprocessing import Process  # TODO delete if isn't used
import threading

RELATIVE_ACC = 1.8
CAMERA_FPS = 30
ARM_DELAY = 1
CROP_SIZE = (160, 480)  # (y,x)
FRAME_SIZE = (480, 640)  # (y,x)
SCREEN_SIZE = (12, 16)  # (y,x)
ACC = RELATIVE_ACC * SCREEN_SIZE[0]
INTEGRATE_WITH_MECHANICS = True

# for hakab
oops = 0
success = 0

on_screen_fruits = []
SIMULATE = False
simulation_queue_lock = threading.Condition()
simulation_thread = None
simulation_queue = []


class Trajectory:
    def __init__(self, x0, y0, v, theta):
        self.x0 = x0
        self.y0 = y0
        self.v = v
        self.theta = theta

    def calc_trajectory(self):
        return lambda t: (x_trajectory(t, self.x0, self.v, self.theta), y_trajectory(t, self.y0, self.v, self.theta))

    def calc_peak(self):
        t = self.v * math.sin(self.theta) / ACC
        return t, self.calc_trajectory()(t)

    def calc_life_time(self):
        t = 2 * self.v * math.sin(self.theta) / ACC
        return t


def update_fruits(fruits):
    fruits_locs = [[pixel2cm(pix_loc) for pix_loc in fruit.centers] for fruit in fruits]
    centers = [[center for center in fruit.centers] for fruit in fruits]
    centers2 = [[cm2pixel(loc) for loc in fruit_locs] for fruit_locs in fruits_locs]
    fruit_trajectories = [get_trajectory(fruit_locs) for fruit_locs in fruits_locs]
    on_screen_fruits.extend([[fruit_trajectories[i], fruits[i].time_created] for i in range(len(fruits))])
    # on_screen_fruits.extend(fruits)
    fruits[:] = []

    # fruit_trajectories = [get_trajectory(fruit_locs) for fruit_locs in fruits_locs]
    # on_screen_fruits.extend([[fruit_trajectories[i], fruits[i].time_created] for i in range(len(fruits))])


def create_slice():
    return calc_slice(on_screen_fruits)


def do_slice(slice):
    parametrization, timer, t_peak, fruits_trajectories = slice
    # time.sleep(time_until_slice(slice))
    if SIMULATE:
        slm.run_simulation(parametrization, fruits_trajectories)
    else:
        ArduinoCommunication.make_slice_by_trajectory(parametrization)


# def create_and_do_slice():
#     global lock
#     #lock.acquire()
#     print(3333333333333333333)
#     gui_lock.acquire()
#     slice = create_slice()
#     p = Process(target=do_slice, args=(slice,))
#     p.start()
#     lock.release()


def update_and_slice(fruits):
    global simulation_queue
    global simulation_queue_lock
    update_fruits(fruits)
    if simulation_queue_lock.acquire(False):
        if len(on_screen_fruits) > 0:
            slice = create_slice()
            simulation_queue.append(slice)
            print("Length of queue : " + str(len(simulation_queue)))
            if (len(simulation_queue) == 2):
                print("x")
        simulation_queue_lock.notify()
        simulation_queue_lock.release()


def pixel2cm(pix_loc):
    '''
    :param pix_loc: a pixel in order (x,y).
    :return: (x coord of screen, y coord of screen) when we look at the screen from the opposite side.
            we look at the opposite side because the arm is looking at the screen from it's top
            and we look at it from the bottom
    '''
    (j_coord_crop, i_coord_crop, t) = pix_loc
    i_coord_frame = FRAME_SIZE[0] - CROP_SIZE[0] + i_coord_crop
    j_coord_frame = FRAME_SIZE[1] / 2 - CROP_SIZE[1] / 2 + j_coord_crop
    i_coord_screen = (float(i_coord_frame / FRAME_SIZE[0])) * SCREEN_SIZE[0]
    j_coord_screen = (1 - float(j_coord_frame / FRAME_SIZE[1])) * SCREEN_SIZE[1]
    return j_coord_screen, i_coord_screen, t  # (x,y)


def cm2pixel(cm_loc):
    """
    :param cm_loc: cm location in order (x,y)
    :return: pixel location in order (y, x, t)
    """
    x_coord_screen, y_coord_screen, t = cm_loc
    x_coord_frame = int(x_coord_screen * float(FRAME_SIZE[1]) / SCREEN_SIZE[1])
    y_coord_frame = int((1.0 - float(y_coord_screen / SCREEN_SIZE[0])) * FRAME_SIZE[0])
    return y_coord_frame, x_coord_frame, t


def rms(array):
    """
    calcs rms
    :return: root mean square
    """
    sum_squares = 0
    length = len(array)
    for i in range(length):
        sum_squares += array[i]**2
    mean = sum_squares / length
    return math.sqrt(mean)


def get_trajectory(fruit_locs):

    time_between_2_frames = 1.0 / CAMERA_FPS

    x_coords = [fruit_loc[0] for fruit_loc in fruit_locs]  # TODO this is a bug. need to make in a loop
    y_coords = [fruit_loc[1] for fruit_loc in fruit_locs]
    t_coords = [fruit_loc[2] for fruit_loc in fruit_locs]

    x_total = x_coords[-1] - x_coords[0]
    y_total = y_coords[-1] - y_coords[0]
    t_total = (len(x_coords) - 1) * time_between_2_frames

    r_coords = [0 for _ in range(len(x_coords) - 1)]
    for i in range(len(y_coords) - 1):
        r_coords[i] = math.sqrt((x_coords[i+1]-x_coords[i])**2 + (y_coords[i+1]-y_coords[i])**2)

    r_total = math.sqrt(x_total**2 + y_total**2)

    x0 = st.mean(x_coords)
    y0 = st.mean(y_coords)
    if x_total == 0:  # to prevent division by zero
        x_total = 0.001

    theta_array = [0 for _ in range(len(x_coords)-1)]
    for i in range(len(x_coords)-1):
        if (x_coords[i+1] - x_coords[i]) != 0:  # to prevent division by zero
            delta_x = (x_coords[i+1] - x_coords[i])
        else:
            delta_x = 0.001
        theta_array[i] = math.pi - math.atan((y_coords[i+1] - y_coords[i]) / delta_x)
    theta_median = st.median(theta_array)  # best theta
    theta_mean = st.mean(theta_array)
    theta_start_to_end = math.pi - math.atan(y_total / x_total)
    theta = theta_median

    r_total_real = abs((SCREEN_SIZE[0] / 3 / math.sin(theta)))  # 3 is because the screen is croped to third
    v_array = [0 for _ in range(len(r_coords))]
    vy_array = [0 for _ in range(len(r_coords))]
    for i in range(len(r_coords)):
        v_array[i] = r_coords[i] / time_between_2_frames
        vy_array[i] = (y_coords[i+1] - y_coords[i]) / time_between_2_frames

    v0_median = st.median(v_array)
    v0_mean = st.mean(v_array)
    v0_rms = rms(v_array)
    vy_median = st.median(vy_array)
    vy_mean = st.mean(vy_array)

    v0_by_vy_end_to_end = y_total / t_total / math.sin(theta)

    v0_by_vy_mean = vy_mean / math.sin(theta)

    v0_by_vy_median = vy_median / math.sin(theta)

    v0_stupid = r_total_real / (10 * time_between_2_frames)

    v0_start_to_end = r_total / t_total

    v0 = v0_mean

    # if len(x_coords) < 10:
    #     v0 = r_total_real / (8 * time_between_2_frames)
    # else:
    #     v0 = r_total_real / t_total

    # v0 = r_total_real / t_total

    trajectory = Trajectory(x0, y0, v0, theta)

    # plt.plot(x_coords, y_coords)
    # plt.show()

    # global success
    # global oops
    # try:
    #     popt, pcov = curve_fit(trajectory_physics, x_coords, y_coords)
    #     print(popt[0], popt[1], popt[2])
    #     print("success")
    #     success += 1
    # except:
    #     popt = (0, 0, 0)
    #     print("oops")
    #     oops += 1
    # print("fitted: ", success, "didn't fit: ", oops)
    # x0_par, v_par, theta_par = popt
    # trajectory = Trajectory(x0_par, v_par, theta_par)

    # ----------draw trajectory-------------- #
    T = 3
    dt = 0.1
    times = range(-int(T / dt), int(T / dt))
    xy = [[0 for _ in times], [0 for _ in times]]
    route = trajectory.calc_trajectory()
    for i in times:
        xy[0][i], xy[1][i] = route(dt * i)

    # plt.plot.xlim(left, right)

    plt.plot(xy[0], xy[1], 'ro')
    # plt.show()
    plt.plot(x_coords, y_coords,'bo')
    plt.ylim(0, 13)
    plt.xlim(0, 16)
    # plt.show()
    return trajectory


def trajectory_physics(x, x0, v, theta):
    return SCREEN_SIZE[0] - (x - x0) * math.tan(theta) + ACC * (x - x0) ** 2 / (2 * v ** 2 * math.cos(theta) ** 2)


def x_trajectory(t, x0, v, theta):
    return x0 + v * math.cos(theta) * t


def y_trajectory(t, y0, v, theta):
    # return SCREEN_SIZE[0] - v * math.sin(theta) * t + 0.5 * ACC * t ** 2
    return y0 - v * math.sin(theta) * t + 0.5 * ACC * t ** 2


def calc_slice(fruit_trajectories_and_starting_times):
    # time.sleep(time_until_slice())
    # return SliceTypes.radius_slice(get_arm_loc(), fruit_trajectories_and_starting_times)
    return SliceTypes.radius_slice(get_arm_loc(), fruit_trajectories_and_starting_times)
    # return None, None, None


def get_arm_loc():
    return -SCREEN_SIZE[1]/2+3, 3


def time_until_slice(fruit):
    _, timer, t_peak = fruit
    return timer + t_peak - time.clock()


def init_info(frame_size, crop_size = CROP_SIZE, screen_size = SCREEN_SIZE):
    global CROP_SIZE, FRAME_SIZE, SCREEN_SIZE
    CROP_SIZE = crop_size
    FRAME_SIZE = frame_size
    SCREEN_SIZE = screen_size


def remove_sliced_fruits(fruits):
    # if len(fruits) != len(on_screen_fruits):
    #     print("FUCK")
    for fruit in fruits:
        on_screen_fruits.remove(fruit)
    for fruit in on_screen_fruits:
        traj, timer = fruit
        if time.clock() > timer + traj.calc_life_time():
            on_screen_fruits.remove(fruit)


def simulation_thread_run():
    global simulation_queue_lock
    global simulation_queue
    while True:
        simulation_queue_lock.acquire()
        while len(simulation_queue) == 0:
            simulation_queue_lock.wait()
        slice = simulation_queue[0]
        simulation_queue.remove(slice)
        do_slice(slice)
        simulation_queue_lock.release()


def init_everything():
    if INTEGRATE_WITH_MECHANICS:
        global simulation_thread
        simulation_thread = Thread(target=simulation_thread_run)
        simulation_thread.start()
    else:
        pass


if __name__ == "__main__":
    slice_and_times = create_slice()
    while True:
        do_slice(slice_and_times)
