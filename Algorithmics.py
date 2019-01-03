import math
import matplotlib.pyplot as plt
import statistics as st
import SliceTypes
import time
import ArduinoCommunication
import Simulation as slm
from threading import Thread
import threading

"""
this was SliceCreator. this file is the brain of the algoritmics - it gets data from image prosseccing 
and generates slices (x and y locations) for arduino.
the coordinates here is (begadol) (x,y) when the 0,0 is at bottom left of the flipped screen
(for example, the parabula of fruits routes is smiling).
"""

# ----------------- CONSTANTS -------------------
RELATIVE_ACC = 1.478  # from experiences we did it tracker program
CAMERA_FPS = 30
TIME_BETWEEN_2_FRAMES = 1.0 / CAMERA_FPS
CROP_SIZE = (160, 480)  # (y,x)
FRAME_SIZE = (480, 640)  # (y,x)
SCREEN_SIZE = (12, 16)  # (y,x)
ACC = RELATIVE_ACC * SCREEN_SIZE[0]
INTEGRATE_WITH_MECHANICS = False

on_screen_fruits = []
SIMULATE = False
simulation_queue_lock = threading.Condition()
simulation_thread = None
simulation_queue = []


# ------------- CONVERTING FUNCTIONS -------------
def pixel2cm(pix_loc):
    """
    :param pix_loc: a pixel in order (x,y).
    :return: (x coord of screen, y coord of screen) when we look at the screen from the opposite side.
            we look at the opposite side because the arm is looking at the screen from it's top
            and we look at it from the bottom
    """
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


# --------------- TRAJECTORY CLASS ---------------
class Trajectory:
    """
    class of fruit trajectory
    """

    def __init__(self, x0, y0, v, theta):
        """
        initiates the parameters for the trajectory
        :param x0: by the formula
        :param y0: by the formula
        :param v: by the formula
        :param theta: by the formula
        """
        self.x0 = x0
        self.y0 = y0
        self.v = v
        self.theta = theta

    def calc_trajectory(self):
        """
        return the trajectory of the fruit by the its parameters
        :return: a function of (x, y) by t (0 is the time of (x0, y0) and calc_life_time() is the time of the end of
        the trajectory (not exactly the end))
        """

        def get_xy_by_t(t):
            """
            :param t: double time
            :return: tuple (x, y) in cm
            """
            x = self.x_trajectory(t, self.x0, self.v, self.theta)
            y = self.y_trajectory(t, self.y0, self.v, self.theta)
            return x, y

        return get_xy_by_t

    def calc_peak(self):
        """
        calculates the time and the y of the peak
        :return: tuple (t, y) in sec, cm
        """
        t = self.v * math.sin(self.theta) / ACC
        return t, self.calc_trajectory()(t)

    def calc_life_time(self):
        """
        returns the time that the fruit gets to the symmetric location to the initial location (x0, y0)
        :return: double time in sec
        """
        t = 2 * self.v * math.sin(self.theta) / ACC
        return t

    def x_trajectory(self, t, x0, v, theta):
        """
        returns the x value according to the formula of free fall
        :param x0: by the formula
        :param y0: by the formula
        :param v: by the formula
        :param theta: by the formula
        :return: x value according to the formula of free fall in cm
        """
        return x0 + v * math.cos(theta) * t

    def y_trajectory(self, t, y0, v, theta):
        """
        returns the y value according to the formula of free fall
        :param x0: by the formula
        :param y0: by the formula
        :param v: by the formula
        :param theta: by the formula
        :return: y value according to the formula of free fall in cm
        """
        # return SCREEN_SIZE[0] - v * math.sin(theta) * t + 0.5 * ACC * t ** 2
        return y0 - v * math.sin(theta) * t + 0.5 * ACC * t ** 2


# -------------------- slicing functions ------------------
def update_fruits(fruits):
    """
    ron and eran have to explain
    :param fruits:
    :return:
    """
    fruits_locs = [[pixel2cm(pix_loc) for pix_loc in fruit.centers] for fruit in fruits]
    # centers = [[center for center in fruit.centers] for fruit in fruits]
    # centers2 = [[cm2pixel(loc) for loc in fruit_locs] for fruit_locs in fruits_locs]
    fruit_trajectories = [get_trajectory_by_fruit_locations(fruit_locs) for fruit_locs in fruits_locs]
    on_screen_fruits.extend([[fruit_trajectories[i], fruits[i].time_created] for i in range(len(fruits))])
    # on_screen_fruits.extend(fruits)
    fruits[:] = []

    # fruit_trajectories = [get_trajectory(fruit_locs) for fruit_locs in fruits_locs]
    # on_screen_fruits.extend([[fruit_trajectories[i], fruits[i].time_created] for i in range(len(fruits))])


def create_slice():
    """
    returns a slice according to the fruits that are on the screen
    :return: tuple of (slice_trajectory, timer, t_peak, fruit_trajectories_and_starting_times)
    """
    return calc_slice(on_screen_fruits)


def do_slice(slice_trajectory):
    """
    activate the simulation or the arduino by the given trajectory
    :param slice_trajectory: function of the location (x, y) of the pen in cm
    """
    parametrization, timer, t_peak, fruits_trajectories = slice_trajectory
    # run simulation
    if SIMULATE:
        slm.run_simulation(parametrization, fruits_trajectories)
    # run arduino
    else:
        ArduinoCommunication.make_slice_by_trajectory(parametrization)


def update_and_slice(fruits):
    """
    ron and eran need to explain
    :param fruits:
    :return:
    """
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


def rms(array):
    """
    calculates rms
    :return: root mean square
    """
    sum_squares = 0
    length = len(array)
    for i in range(length):
        sum_squares += array[i] ** 2
    mean = sum_squares / length
    return math.sqrt(mean)


def get_r_coords_by_xy_coords(x_coords, y_coords):
    """

    :param x_coords:
    :param y_coords:
    :return:
    """
    r_coords = [0 for _ in range(len(x_coords) - 1)]
    for i in range(len(y_coords) - 1):
        r_coords[i] = math.sqrt((x_coords[i + 1] - x_coords[i]) ** 2 + (y_coords[i + 1] - y_coords[i]) ** 2)
        return r_coords


def get_trajectory_by_fruit_locations(fruit_locs):
    """
    creating a trajectory according to the locations of the fruit by fitting speed v0 and initial angle theta
    :param fruit_locs: 2d list of [x, y, t] of the fruit (locations of 1 fruit)
    :return: trajectory object with the fitted values for speed (v0) and angle (theta)
    """

    x_coords = [fruit_loc[0] for fruit_loc in fruit_locs]
    y_coords = [fruit_loc[1] for fruit_loc in fruit_locs]
    t_coords = [fruit_loc[2] for fruit_loc in fruit_locs]  # times from image processing are not accurate for sure
    r_coords = get_r_coords_by_xy_coords(x_coords, y_coords)

    # plot the given fruit locations (not the trajectory)
    # plt.plot(x_coords, y_coords)
    # plt.show()

    # values between last location to first location
    x_total = x_coords[-1] - x_coords[0]
    y_total = y_coords[-1] - y_coords[0]
    t_total = (len(x_coords) - 1) * TIME_BETWEEN_2_FRAMES
    r_total = math.sqrt(x_total ** 2 + y_total ** 2)

    # *****options for x0 and y0 values*****
    x0_mean = st.mean(x_coords) if x_total != 0 else 0.001  # to prevent division by zero
    y0_mean = st.mean(y_coords)
    x0_last = x_coords[-1]
    y0_last = y_coords[-1]
    x0 = x0_mean
    y0 = y0_mean

    # *****options for theta value*****
    theta_array = [0 for _ in range(len(x_coords) - 1)]
    for i in range(len(x_coords) - 1):
        if (x_coords[i + 1] - x_coords[i]) != 0:  # to prevent division by zero
            delta_x = (x_coords[i + 1] - x_coords[i])
        else:
            delta_x = 0.001
        theta_array[i] = math.pi - math.atan((y_coords[i + 1] - y_coords[i]) / delta_x)
    theta_median = st.median(theta_array)  # best theta
    theta_mean = st.mean(theta_array)
    theta_start_to_end = math.pi - math.atan(y_total / x_total)
    theta = theta_median

    # *****options for v0 value*****
    r_total_real = abs((SCREEN_SIZE[0] / 3 / math.sin(theta)))  # 3 is because the screen is croped to third
    v_array = [0 for _ in range(len(r_coords))]
    vy_array = [0 for _ in range(len(r_coords))]
    for i in range(len(r_coords)):
        v_array[i] = r_coords[i] / TIME_BETWEEN_2_FRAMES
        vy_array[i] = (y_coords[i + 1] - y_coords[i]) / TIME_BETWEEN_2_FRAMES

    v0_median = st.median(v_array)
    v0_mean = st.mean(v_array)
    v0_rms = rms(v_array)
    vy_median = st.median(vy_array)
    vy_mean = st.mean(vy_array)

    v0_by_vy_end_to_end = y_total / t_total / math.sin(theta)
    v0_by_vy_mean = vy_mean / math.sin(theta)
    v0_by_vy_median = vy_median / math.sin(theta)
    v0_stupid = r_total_real / (10 * TIME_BETWEEN_2_FRAMES)
    v0_start_to_end = r_total / t_total

    v0 = v0_mean

    # the trajectory by the fitted values (the returned object)
    trajectory = Trajectory(x0, y0, v0, theta)

    # plot the trajectory
    draw_trajectory_matplotlib(trajectory, x_coords, y_coords)

    return trajectory


def draw_trajectory_matplotlib(trajectory, x_coords, y_coords):
    """
    plot the trajectory and the fruit locations
    :param trajectory: trajectory object
    :param x_coords: x values of 1 fruit
    :param y_coords: y values of 1 fruit
    """
    T = 3
    dt = 0.1
    x_lim = 13
    y_lim = 16

    times = range(-int(T / dt), int(T / dt))
    xy = [[0 for _ in times], [0 for _ in times]]
    route = trajectory.calc_trajectory()
    for i in times:
        xy[0][i], xy[1][i] = route(dt * i)

    plt.plot(xy[0], xy[1], 'ro')
    plt.plot(x_coords, y_coords, 'bo')
    plt.ylim(0, x_lim)
    plt.xlim(0, y_lim)
    plt.show()


def calc_slice(fruit_trajectories_and_starting_times):
    """
    return the chosen slice - function of (x, y) by t
    :param fruit_trajectories_and_starting_times:
    :return:
    """
    return SliceTypes.radius_slice(get_pen_loc(), fruit_trajectories_and_starting_times)


def get_pen_loc():
    """
    :return: the location of the pen (x, y) in cm
    """
    # location (3cm, 3cm) from the bottom-left corner
    x_location = -SCREEN_SIZE[1] / 2 + 3
    y_location = 3
    return x_location, y_location


def init_info(frame_size, crop_size=CROP_SIZE, screen_size=SCREEN_SIZE):
    """
    eran and ron need to explain
    :param frame_size:
    :param crop_size:
    :param screen_size:
    :return:
    """
    global CROP_SIZE, FRAME_SIZE, SCREEN_SIZE
    CROP_SIZE = crop_size
    FRAME_SIZE = frame_size
    SCREEN_SIZE = screen_size


def remove_sliced_fruits(fruits):
    """
    eran and ron need to explain
    :param fruits:
    :return:
    """
    # if len(fruits) != len(on_screen_fruits):
    #     print("FUCK")
    for fruit in fruits:
        on_screen_fruits.remove(fruit)
    for fruit in on_screen_fruits:
        traj, timer = fruit
        if time.clock() > timer + traj.calc_life_time():
            on_screen_fruits.remove(fruit)


def simulation_thread_run():
    """
    eran and ron need to explain
    :return:
    """
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
    """
    eran and ron need to explain
    :return:
    """
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
