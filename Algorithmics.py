"""
Previously named SliceCreator. This file is the brain of the algorithmic module - it gets data from image processing
and generates slices (x and y locations) for arduino.
the coordinates here is (generally speaking) (x,y) when the 0,0 is at bottom left of the flipped screen
(the parabola of fruits routes is smiling).
"""

import math
import statistics as st
import SliceTypes
import time
import ArduinoCommunication4 as Ac
import Simulation as Sim
from threading import Thread
import threading
import numpy as np
import itertools

# ----------------- CONSTANTS -------------------
PIXELS_PER_CM = 40
MULTI = False

# first acc is measured, second is from fazkanoot
# RELATIVE_ACC = 1.478  # from experiences we did it tracker program
PART_OF_SCREEN_FOR_IP = 0.0
RELATIVE_ACC = 1.4   # not from experiences we did it tracker program
CAMERA_FPS = 30  # frames per second
TIME_BETWEEN_2_FRAMES = 1.0 / CAMERA_FPS  # in sec
FRAME_SIZE = (480, 640)  # (y,x) in pixels
CROP_SIZE = (160, 480)  # (y,x) in pixels
SCREEN_SIZE = (12.0, 16.0)  # (y,x) in cm
FULL_SCREEN = (12.0, 16.0)
DISTANCE_FROM_TABLET = Ac.d
ARM_LOC_BEGINNING_ALGO = (1.0, 4.0)
ARM_LOC_DOCKING = (15.0, 4.0)
ACC = RELATIVE_ACC * SCREEN_SIZE[0]
INTEGRATE_WITH_MECHANICS = True  # make True to send slices to ArduinoCommunication

LINEAR = 0
RADIUS = 1
THROUGH_POINTS = 2
SLICE_TYPE = LINEAR

SLICE_QUALITY_FACTOR_THRESH = 0
MINIMAL_NUMBER_OF_FRUITS_FOR_COMBO = 3
MAX_TIME_FOR_COMBO = 400  # in ms

# on_screen_fruits = []
SIMULATE = True  # make True to activate simulation
slice_queue_lock = threading.Condition()
simulation_thread = None
slice_queue = []
during_slice = False
DOCKING = True

INITIALIZED = False

VY_MAX = 24
VY_MIN = 7
VX_MAX = 5


# ------------- CONVERTING FUNCTIONS -------------
def pixel2cm(pix_loc):
    """
    :param pix_loc: a pixel in order (x,y).
    :return: (x coord of screen, y coord of screen) when we look at the screen from the opposite side.
            we look at the opposite side because the arm is looking at the screen from it's top
            and we look at it from the bottom
    """
    (j_coord_frame, i_coord_frame, t) = crop2frame(pix_loc)
    i_coord_screen = (float(i_coord_frame / FRAME_SIZE[0])) * SCREEN_SIZE[0]
    j_coord_screen = (1 - float(j_coord_frame / FRAME_SIZE[1])) * SCREEN_SIZE[1]
    return j_coord_screen, i_coord_screen, t  # (x,y,t)


def crop2frame(pix_loc):
    (j_coord_crop, i_coord_crop, t) = pix_loc
    i_coord_frame = FRAME_SIZE[0] + (- CROP_SIZE[0] + i_coord_crop)
    j_coord_frame = FRAME_SIZE[1] / 2 - CROP_SIZE[1] / 2 + j_coord_crop
    return int(j_coord_frame), int(i_coord_frame), t


def cm2pixel(cm_loc):
    """
    :param cm_loc: cm location in order (x, y, t)
    :return: pixel location in order (y, x, t)
    """
    x_coord_screen, y_coord_screen, t = cm_loc
    x_coord_frame = int(x_coord_screen * (float(FRAME_SIZE[1]) / SCREEN_SIZE[1]))
    x_coord_frame = FRAME_SIZE[1] - x_coord_frame
    y_coord_frame = int((float(y_coord_screen / SCREEN_SIZE[0])) * FRAME_SIZE[0])
    return y_coord_frame, x_coord_frame, t


# --------------- TRAJECTORY CLASS ---------------
class Trajectory:
    """
    class of fruit trajectory
    """

    def __init__(self, x0, y0, v0x, v0y):
        """
        initiates the parameters for the trajectory
        """
        self.x0 = x0
        self.y0 = y0
        self.v0x = v0x
        self.v0y = v0y

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
            return self.x_trajectory(t), self.y_trajectory(t)

        return get_xy_by_t

    def calc_peak(self):
        """
        calculates the time and the y of the peak
        :return: tuple (t, y) in sec, cm
        """
        t = self.v0y / ACC
        return t, self.calc_trajectory()(t)

    def calc_life_time(self):  # TODO improve function to return the time until the fruit exits the screen
        """
        returns the time that the fruit gets to the symmetric location to the initial location (x0, y0)
        :return: double time in sec
        """
        t = 2 * abs(self.v0y) / ACC
        return t

    def x_trajectory(self, t):
        """
        returns the x value according to the formula of free fall
        :param t: time
        :return: x value according to the formula of free fall in cm
        """
        return self.x0 + self.v0x * t

    def y_trajectory(self, t):
        """
        returns the y value according to the formula of free fall
        :param t: time
        :return: y value according to the formula of free fall in cm
        """
        # return SCREEN_SIZE[0] - v * math.sin(theta) * t + 0.5 * ACC * t ** 2
        return self.y0 + self.v0y * t + 0.5 * ACC * t ** 2


# -------------------- slicing functions ------------------
# def update_fruits(fruits):
#     """
#     Updates on_screen_fruits according to fruits list acquired from image processing.
#     :param fruits: list of Fruit objects - fruits to add to on_scree_fruits
#     """
#     fruits_locs = [[pixel2cm(pix_loc[0]) for pix_loc in fruit.centers] for fruit in fruits]
#     # centers = [[center for center in fruit.centers] for fruit in fruits]
#     # centers2 = [[cm2pixel(loc) for loc in fruit_locs] for fruit_locs in fruits_locs]
#     fruit_trajectories = [get_trajectory_by_fruit_locations(fruit_locs) for fruit_locs in fruits_locs]
#     on_screen_fruits.extend([[fruit_trajectories[i], fruits[i].time_created] for i in range(len(fruits))])
#     # on_screen_fruits.extend(fruits)
#     fruits[:] = []
#     # fruit_trajectories = [get_trajectory(fruit_locs) for fruit_locs in fruits_locs]
#     # on_screen_fruits.extend([[fruit_trajectories[i], fruits[i].time_created] for i in range(len(fruits))])


def create_slice(state, time_to_slice):
    """
    Returns the optimal slice according to the fruits that are on the screen and the given time to slice.
    :param state: the state of the game
    :param time_to_slice: the desired time to execute the slice.
    :return: tuple of (xy_points,  sliced_fruits)
    """

    fruits_and_locs = [(fruit,loc) for (fruit, loc) in state.get_fruits_locations(time_to_slice, state.fruits_in_range)
                       if not point_out_of_screen_or_bound_algo_coords(loc)]
    critical_fruits_locs = state.get_fruits_locations(time_to_slice, state.get_critical_fruits())
    arm_loc = state.arm_loc
    docking = state.docking
    # arm_loc_algo_coordinates = mech_to_algo(arm_loc)
    ordered_fruits_and_locs = order_fruits_and_locs(arm_loc, fruits_and_locs)
    xy_points_to_go_through, sliced_fruits = create_best_slice(state.arm_loc, ordered_fruits_and_locs,
                                                               critical_fruits_locs, docking)
    return xy_points_to_go_through, sliced_fruits


def order_fruits_and_locs(arm_loc, fruits_and_locs):
    return sorted(fruits_and_locs, key=key_theta(arm_loc))


def create_best_slice(arm_loc, ordered_fruits_and_locs, critical_fruits_locs, docking):
    """
    creates the best slice it can with the given fruits on screen.
    :param arm_loc: the location of the arm initially.
    :param ordered_fruits_and_locs: list of (fruit, loc) sorted by distance on x axis from the arm loc.
    :param critical_fruits_locs: fruits about to be lose.
    :param docking: point in (x,y) for final location of arm (enter empty tuple for no docking)
    :return: points to slice through and the fruits sliced.
    """
    current_arm_loc = arm_loc
    slice_points = []
    sliced_fruits = []
    for fruits_and_locs in combinations_of_elements(ordered_fruits_and_locs):
        locs = [loc for (fruit, loc) in fruits_and_locs]
        temp_slice_points = calc_slice(arm_loc, locs, tuple())
        if good_slice(temp_slice_points):
            slice_points.extend(temp_slice_points)
            sliced_fruits.extend([fruit for (fruit, loc) in fruits_and_locs])
            current_arm_loc = temp_slice_points[-1]
            break
    if SLICE_TYPE == THROUGH_POINTS:
        critical_fruits_not_sliced = order_fruits_and_locs(current_arm_loc,
                                                           [(fruit, loc) for (fruit, loc) in critical_fruits_locs if
                                                            fruit not in sliced_fruits])
        if critical_fruits_not_sliced:
            critical_fruits_not_sliced_locs = [loc for (fruit, loc) in critical_fruits_not_sliced]
            remaining_slice = calc_slice(current_arm_loc, critical_fruits_not_sliced_locs, tuple())
            if slice_points:
                slice_points.extend(remaining_slice[1:])
            else:
                slice_points.append(remaining_slice)
            sliced_fruits.extend([fruit for (fruit, loc) in critical_fruits_not_sliced])
    if DOCKING:
        slice_points.append(algo_to_mech(docking))
    return slice_points, sliced_fruits


def combinations_of_elements(s):
    """
    Returns the combinations of elements to slice in order to get combo
    :param s: fruits and locations of all fruits in range for slice.
    :return: subgroups of fruits larger than number of fruits needed for combo by
     descending order by size.
    """
    if len(s) < MINIMAL_NUMBER_OF_FRUITS_FOR_COMBO:
        return [s]
    return list(itertools.chain.from_iterable(itertools.combinations(s, r)
                                              for r in range(len(s), MINIMAL_NUMBER_OF_FRUITS_FOR_COMBO - 1, -1)))


def time_for_slice(points):
    if MULTI:
        points = flip_points_for_multiplayer(points)
    steps_theta, steps_phi = Ac.generate_steps_from_points(points)
    return Ac.calc_time_of_slice(steps_theta, steps_phi)


def good_slice(points_of_slice_to_evaluate):
    """
    Determines whether or not a slice is "good enough" (creates combo).
    :param points_of_slice_to_evaluate: the slice which we want to test.
    :return: true if the slice should be done.
    """
    # return time_for_slice(points_of_slice_to_evaluate) < MAX_TIME_FOR_COMBO
    return True


def key_x(arm_loc):
    def distance_from_arm_in_x(fruit_and_loc):
        _, loc = fruit_and_loc
        return abs(loc[0] - arm_loc[0])

    return distance_from_arm_in_x

def key_theta(arm_loc):
    def distance_from_arm_in_x(fruit_and_loc):
        fruit, loc = fruit_and_loc
        return abs(Ac.xy2angles(algo_to_mech(loc))[0] - Ac.xy2angles(algo_to_mech(arm_loc))[0])

    return distance_from_arm_in_x



def key_theta(arm_loc):
    def distance_from_arm_in_theta(fruit_and_loc):
        _, loc = fruit_and_loc
        return abs(Ac.xy2angles(algo_to_mech(loc))[0] - Ac.xy2angles(algo_to_mech(arm_loc))[0])

    return distance_from_arm_in_theta


def flip_points_for_multiplayer(points):
    points = [flip_point_for_multiplayer(p) for p in points]
    print("after flip: ", points)
    return points

def flip_point_for_multiplayer(p):
    return SCREEN_SIZE[1] - p[0], SCREEN_SIZE[0] - p[1]


def do_slice(points_to_slice, sliced_fruits):
    """
    Activate the simulation or the arduino by the given slice.
    :param points_to_slice: list of points the slice should go through
    :param sliced_fruits: fruits that the slice is supposed to cut (for simulation)
    """
    # if MULTI:
    #     points_to_slice = flip_points_for_multiplayer(points_to_slice)
    parametrization = points_to_slice
    print(parametrization)
    if points_out_of_screen_or_bound(parametrization):
        print("doing linear slice")
        parametrization = SliceTypes.linear_slice(algo_to_mech(get_pen_loc()[0]), [])
    # time_to_slice = 0
    # run simulation
    if SIMULATE:
        Sim.run_simulation(parametrization, sliced_fruits)
    else:
        Ac.make_slice_by_trajectory(parametrization)

def points_out_of_screen_or_bound(points):
    for point in points:
        if point_out_of_screen(point) or point_out_of_bound(point):
            return True
    return False

def point_out_of_screen_or_bound_algo_coords(point):
    if point_out_of_screen_algo(point) or point_out_of_bound_algo(point):
        return True


def point_out_of_screen(point):
    (x, y) = point
    return x_out_of_bound(x) or y_out_of_bound(y)

def point_out_of_screen_algo(point):
    x, y = point
    return not(0<x<SCREEN_SIZE[1] and 0<y<SCREEN_SIZE[0])

def point_out_of_bound(point):
    return SliceTypes.distance(point, SliceTypes.ORIGIN) >=  SliceTypes.get_max_dis()


def point_out_of_bound_algo(point):
    x,y=point
    return SliceTypes.distance(point, (SCREEN_SIZE[1]/2, FULL_SCREEN[0] - SCREEN_SIZE[0] + SliceTypes.d)) >=\
           SliceTypes.get_max_dis()



def x_out_of_bound(x):
    return not -SCREEN_SIZE[1] / 2 < x < SCREEN_SIZE[1] / 2

def y_out_of_bound(y):
    return not 0 < y < SCREEN_SIZE[0]


def add_slice_to_queue(slice_to_add, sliced_fruits):
    """
    Updates the fruits on the screen and than creates a slice and adds it to slice queue.
    :param slice_to_add: slice to add to queue
    :param sliced_fruits: The fruits extracted by the image processing module
    """
    global slice_queue
    global slice_queue_lock
    # update_fruits(fruits)  # Updates the fruits on screen with the fruits extracted by the image processing module.
    if slice_queue_lock.acquire(False):  # If the access to slice queue is available, it means we are not in the middle
        # of a slice and we can create a new one.
        # if len(on_screen_fruits) > 0:  # If there are fruits on the screen we want to create a slice.
        #     new_slice = create_slice()
        slice_queue.append((slice_to_add, sliced_fruits))  # Add the new slice to slice queue.
        slice_queue_lock.notify()
        slice_queue_lock.release()  # Release the lock on the slice queue.


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
    Calculates the distance between points on trajectory
    :param x_coords: x coordinates along trajectory
    :param y_coords: y coordinates along trajectory
    :return: list of length len(x_coords)-1 of the distances between points
    """
    r_coords = [0 for _ in range(len(x_coords) - 1)]
    for i in range(len(y_coords) - 1):
        r_coords[i] = math.sqrt((x_coords[i + 1] - x_coords[i]) ** 2 + (y_coords[i + 1] - y_coords[i]) ** 2)
    return r_coords


def get_trajectory_by_fruit_locations(fruit_locs):
    """
    creating a trajectory according to the locations of the fruit by fitting speed v0 and initial angle theta
    :param fruit_locs: 2d list of [x, y, t,correlation] of the fruit (locations of 1 fruit)
    :return: trajectory object with the fitted values for speed (v0) and angle (theta)
    """

    x_coords = [fruit_loc[0] for fruit_loc in fruit_locs]
    y_coords = [fruit_loc[1] for fruit_loc in fruit_locs]
    r_coords = get_r_coords_by_xy_coords(x_coords, y_coords)

    # values between last location to first location
    # x_total = x_coords[-1] - x_coords[0] if x_coords[-1] - x_coords[0] != 0 else 0.00001
    # y_total = y_coords[-1] - y_coords[0]
    # t_total = (len(x_coords) - 1) * TIME_BETWEEN_2_FRAMES
    # r_total = math.sqrt(x_total ** 2 + y_total ** 2)

    # *****options for x0 and y0 values*****
    # x0_mean = st.mean(x_coords) if x_total != 0 else 0.001  # to prevent division by zero
    # y0_mean = st.mean(y_coords)
    # x0_last = x_coords[-1]
    # y0_last = y_coords[-1]
    # x0 = x_coords[0]
    # y0 = y_coords[0]

    # *****options for v0 value*****
    # r_total_real = abs((SCREEN_SIZE[0] / 3 / math.sin(theta)))  # 3 is because the screen is croped to third
    # v_array = [0 for _ in range(len(r_coords))]
    if not r_coords:
        raise (Exception("not enough data"))
    vy_array = [0 for _ in range(len(r_coords))]
    vx_array = [0 for _ in range(len(r_coords))]
    r_mean = st.mean(r_coords)
    r_std = np.std(r_coords)
    r_fixed = []
    for i in range(len(r_coords)):
        if abs(r_coords[i] - r_mean) < abs(r_std):
            r_fixed.append(r_coords[i])
    # if r_fixed:
    #     r_mean_fixed = st.mean(r_fixed)
    # else:
    #     r_mean_fixed = 0
    times_with_fix = []
    for i in range(len(r_coords)):
        # time_with_fix = fruit_locs[i+1][2] - fruit_locs[i][2]
        times_with_fix.append(TIME_BETWEEN_2_FRAMES)
        # if abs(r_coords[i] - r_mean) > abs(r_std) and r_mean_fixed != 0:
        #     times_with_fix[i] = TIME_BETWEEN_2_FRAMES * (r_coords[i] / r_mean_fixed)
        # print("time with fix: ", times_with_fix[i])

        # v_array[i] = r_coords[i] / TIME_BETWEEN_2_FRAMES
        time_fixed = times_with_fix[i]
        if time_fixed == 0:
            time_fixed = 0.0000001
        vx_array[i] = (x_coords[i + 1] - x_coords[i]) / time_fixed

        cur_vy = (y_coords[i + 1] - y_coords[i]) / time_fixed
        # now try to get the v_entry from current v
        vy = cur_vy - ACC * i * times_with_fix[i]
        vy_array[i] = vy

    # v0_median = st.median(v_array)
    # v0_mean = st.mean(v_array)
    # v0_rms = rms(v_array)
    # vy_median = st.median(vy_array)2222222222222
    vy_mean = st.mean(vy_array)
    vx_mean = st.mean(vx_array)
    # fix the values of the velocities by threshold values: VY_MAX, VY_MIN, VX_MAX
    vx_mean, vy_mean = fix_v_values_by_threshold(vx_mean, vy_mean)
    # v0_mean = math.sqrt(vy_mean ** 2 + vx_mean ** 2)

    # v0_by_vy_end_to_end = y_total / t_total / math.sin(theta)
    # v0_by_vy_mean = vy_mean / math.sin(theta)
    # v0_by_vy_median = vy_median / math.sin(theta)
    # v0_stupid = r_total_real / (10 * TIME_BETWEEN_2_FRAMES)
    # v0_start_to_end = r_total / t_total

    # v0 = v0_mean

    # *****options for theta value*****
    # theta_array = [0 for _ in range(len(x_coords) - 1)]
    # for i in range(len(x_coords) - 1):
    #     if (x_coords[i + 1] - x_coords[i]) != 0:  # to prevent division by zero
    #         delta_x = (x_coords[i + 1] - x_coords[i])
    #     else:
    #         delta_x = 0.001
    #     # theta_array[i] = math.pi - math.atan((y_coords[i + 1] - y_coords[i]) / delta_x)
    #     if vx_mean == 0: vx_mean = 0.00000001
    #     theta_array[i] = -1 * (math.pi - math.atan2(vy_mean, vx_mean))

    # theta_median = st.median(theta_array)  # best theta
    # theta_mean = math.atan(vy_mean/vx_mean)
    # # theta_start_to_end = math.pi - math.atan(y_total / x_total)
    # theta = theta_mean

    # ***** more options for x0 and y0 values*****
    x0_array, y0_array = [], []
    for i in range(len(x_coords) - 1):
        x0_array.append(x_coords[i] - times_with_fix[i] * i * vx_mean)
        y0_array.append(y_coords[i] - times_with_fix[i] * i * vy_mean)

    x0 = st.mean(x0_array)
    y0 = st.mean(y0_array)

    v0y = vy_mean
    v0x = vx_mean

    # the trajectory by the fitted values (the returned object)
    trajectory = Trajectory(x0, y0, v0x, v0y)

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
    t_tot = 3
    dt = 0.1
    # x_lim = 20
    # y_lim = 25

    times = range(-int(t_tot / dt), int(t_tot / dt))
    xy = [[0 for _ in times], [0 for _ in times]]
    route = trajectory.calc_trajectory()
    for i in times:
        xy[0][i], xy[1][i] = route(dt * i)

        # plt.plot(xy[0], xy[1], 'ro')
        # plt.plot(x_coords, y_coords, 'bo')
        # plt.ylim(0, y_lim)
        # plt.xlim(0, x_lim)
        # plt.show()


def algo_to_mech(point):
    """
    Converts a point form the algorithmics and trajectory coordinates to mechanics and Arduino coordinates
    :param point: (x,y) in algorithmics and trajectory coordinates
    :return: (x,y) in mechanics and Arduino coordinates
    """
    if MULTI:
        return Ac.DIMS[0] / 2 - point[0], point[1]
    return Ac.DIMS[0] / 2 - point[0], point[1] + (Ac.DIMS[1] - SCREEN_SIZE[0])


def mech_to_algo(point):
    """
    Converts a point form the mechanics and arduino coordinates to algorithmics and trajectory coordinates
    :param point: (x,y) in mechanics and Arduino coordinates
    :return: (x,y) in algorithmics and trajectory coordinates
    """
    return Ac.DIMS[0] / 2 - point[0], point[1] - (Ac.DIMS[1] - SCREEN_SIZE[0])


def calc_slice(arm_loc, points, docking):
    """
    Calculate the slice that goes through the given points.
    :param arm_loc: location of arm at beginning of slice
    :param points: points the slice should go through in (x,y)
    :param docking: point in (x,y) for final location of arm (enter empty tuple for no docking)
    :return: calculated slice as parametrization, timer, time_to_slice
    """
    if MULTI:
        arm_loc = flip_point_for_multiplayer(arm_loc)
        docking = flip_point_for_multiplayer(docking)
        points = flip_points_for_multiplayer(points)
    points = [algo_to_mech(point) for point in points]
    if docking != ():
        points += [algo_to_mech(docking)]
    mech_arm_loc = algo_to_mech(arm_loc)
    if SLICE_TYPE == LINEAR:
        return SliceTypes.linear_slice(mech_arm_loc, points)
    elif SLICE_TYPE == RADIUS:
        return SliceTypes.radius_slice(mech_arm_loc, points)
    elif SLICE_TYPE == THROUGH_POINTS:
        return SliceTypes.slice_through_fruits(mech_arm_loc, points)


def get_pen_loc():
    """
    :return: the location of the pen (x, y) in mechanics coordinates in cm
    """
    # location (16cm, 4cm) from the bottom-left corner
    x_location = ARM_LOC_BEGINNING_ALGO[0]
    y_location = SCREEN_SIZE[0] + ARM_LOC_BEGINNING_ALGO[1] - FULL_SCREEN[0]
    x_docking = ARM_LOC_DOCKING[0]
    y_docking = SCREEN_SIZE[0] + ARM_LOC_DOCKING[1] - FULL_SCREEN[0]
    # if MULTI:
    #     x_location = -1 * x_location
    #     # y_location = y_location
    #     x_docking = -1 * x_docking
    if DOCKING:
        return (x_location, y_location), (x_docking, y_docking)
    else:
        return (x_location, y_location), ()
    # return -SCREEN_SIZE[1]/2+3, 3  # location (3cm, 3cm) from the bottom-left corner


def time_until_peak(time_created, time_of_peak):
    """
    :param time_created: value of perf_counter when fruit was detected
    :param time_of_peak: time in secs for fruit to reach its peak since detection
    :return: the time until the fruit reaches its peak in secs
    """
    return time_created + time_of_peak - time.perf_counter()


def time_until_slice(time_created, time_of_slice):
    """
    :param time_created: value of perf_counter when slice was created
    :param time_of_slice: value of perf_counter when the slice should be executed
    :return: the time until the slice should be executed in secs
    """
    return time_until_peak(time_created, time_of_slice)


def init_info(frame_size, screen_size=SCREEN_SIZE):
    """
    Initializes the sizes for the screen so that the algorithmics work properly.
    :param frame_size: size of frame in pixels
    :param screen_size: size of screen in cm
    """
    global CROP_SIZE, FRAME_SIZE, SCREEN_SIZE, INITIALIZED
    INITIALIZED = True
    # CROP_SIZE = (frame_size[0] // 3, int(frame_size[1] * 0.75))
    FRAME_SIZE = frame_size
    SCREEN_SIZE = (frame_size[0] * screen_size[1] / frame_size[1], screen_size[1])
    # SCREEN_SIZE = (frame_size[0]*screen_size[1]/frame_size[1], frame_size[1]*screen_size[0]/frame_size[1])
    # SCREEN_SIZE = (frame_size[0] / PIXELS_PER_CM, frame_size[1] / PIXELS_PER_CM)



def mechanics_thread_run():
    """
    The function which runs in a different thread and executes the slices.
    """
    global slice_queue_lock
    global slice_queue
    global during_slice
    while True:
        # Unlocks access to the slice queue (slices waiting to be done).
        slice_queue_lock.acquire()
        # Waits for a slice to enter the queue.
        while len(slice_queue) == 0:
            slice_queue_lock.wait()
        # Retrieves a slice for the queue.
        next_slice, sliced_fruits = slice_queue[0]
        slice_queue.remove((next_slice, sliced_fruits))
        # Executes slice (still memory not unlocked so that we want start a new slice during the previous one).
        during_slice = True
        try:
            do_slice(next_slice, sliced_fruits)
        except:
            pass
        during_slice = False
        # Release the access to the memory so that we can enter new slices to queue.
        slice_queue_lock.release()


def init_everything(slice_type=SLICE_TYPE, integrate_with_mechanics=INTEGRATE_WITH_MECHANICS,
                    simulate=SIMULATE, multi=False):
    """
    Initializes the algorithmics module - opens a thread for the mechanics module.
    :param slice_type: the strategy we want to use this game
    :param integrate_with_mechanics: boolean that decides weather to integrate with mechanics or not.
    :param simulate: boolean that decides weather to activate simulation or not.
    :param multi: True for multiplayer
    """
    global INTEGRATE_WITH_MECHANICS
    INTEGRATE_WITH_MECHANICS = integrate_with_mechanics
    global SIMULATE
    SIMULATE = simulate
    global SLICE_TYPE
    SLICE_TYPE = slice_type
    global MULTI
    MULTI = multi
    if MULTI:
        init_multi_params()
        SliceTypes.init_multi(MULTI)
        global ARM_LOC_BEGINNING_ALGO
        global ARM_LOC_DOCKING
        ARM_LOC_BEGINNING_ALGO = (1.0, 3.0)
        ARM_LOC_DOCKING = (11.0, 3.0)
    # In case we want to integrate with mechanics (simulation or arduino) we must open a new thread for it.
    if INTEGRATE_WITH_MECHANICS:
        global simulation_thread
        simulation_thread = Thread(target=mechanics_thread_run)
        simulation_thread.start()
    else:
        pass


def init_multi_params():
    global FULL_SCREEN
    global FRAME_SIZE
    global CROP_SIZE
    global SCREEN_SIZE
    global ACC
    Sim.init_multi()
    Ac.init_multi_arduino_communication()
    CROP_SIZE = (106, 360)
    SCREEN_SIZE = (8.0, 12.0)
    FULL_SCREEN = (8.0, 12.0)
    ACC = RELATIVE_ACC * SCREEN_SIZE[0]
    # TODO add VY_MAX and this


def fix_v_values_by_threshold(vx_mean, vy_mean):
    # TODO - determine the correct values.
    if abs(vy_mean) > VY_MAX:
        vy_mean = VY_MAX * sign(vy_mean)
    if abs(vy_mean) < VY_MIN:
        vy_mean = VY_MIN * sign(vy_mean)
    if abs(vx_mean) > VX_MAX:
        vx_mean = VX_MAX * sign(vx_mean)
    return vx_mean, vy_mean


def sign(num):
    if num >= 0:
        return 1
    return -1


def in_range_for_slice(point):
    """
    :param point: in (x,y)
    :return: true if the point is in range for slice (for now it's top half of screen)
    """
    return SliceTypes.in_bound(point)


def on_screen(point):
    """
    :param point: in (x,y)
    :return: true if the point is inside the screen of the tablet
    """
    return SliceTypes.in_bound(point, PART_OF_SCREEN_FOR_IP)


if __name__ == "__main__":
    slice_and_times = SliceTypes.linear_slice(get_pen_loc(), [])
    while True:
        do_slice(slice_and_times, [])
