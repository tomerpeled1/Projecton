import math
import Algorithmics

#
LINE_LENGTH = 10

r = 9.9
R = 14.9
d = 17.9
SCREEN = [16, 12]


def tuple_add(tup1, tup2):
    return tup1[0] + tup2[0], tup1[1] + tup2[1]


def tuple_mul(scalar, tup):
    return scalar * tup[0], scalar * tup[1]

def x_algo_to_mech(x_algorithmics):
    """
    :param x_algorithmics: x coordinate in the algorithmics (sliceCreator) coordinate system: (0, 0) on bottom left
    :return: x coordinate in the Mechanics and simulation coordinate system: (0, 0) on bottom middle
            (where the base of the motor is).
    """
    return x_algorithmics - SCREEN[0]/2

def slice_to_peak(arm_loc, fruit_trajectories_and_starting_times):
    """
    make straight line to the peak
    :param arm_loc: (x,y) of arm location
    :param fruit_trajectories_and_starting_times: function of (x,y) by t
    :return: slice, timer, t_peak, fruit_trajectories_and_starting_times
    """
    x_arm_loc, y_arm_loc = arm_loc
    if not fruit_trajectories_and_starting_times:  # if the list of the fruits is empty returns radius_slice
        return radius_slice(arm_loc, fruit_trajectories_and_starting_times)
    chosen_fruit = fruit_trajectories_and_starting_times[0]
    chosen_fruits = [chosen_fruit]
    chosen_trajectory, timer = chosen_fruit
    t_peak, (x_peak, y_peak) = chosen_trajectory.calc_peak()
    x_peak = x_algo_to_mech(x_peak)

    def slice(t):
        x_slice = x_arm_loc + (x_peak - x_arm_loc) * t * 2
        y_slice = y_arm_loc + (y_peak - y_arm_loc) * t * 2
        return x_slice, y_slice

    fruit_trajectories_and_starting_times_copy = fruit_trajectories_and_starting_times.copy()
    Algorithmics.remove_sliced_fruits(chosen_fruits)
    return slice, timer, t_peak, fruit_trajectories_and_starting_times_copy


def stupid_slice(arm_loc, fruit_trajectories_and_starting_times):
    SliceCreator.remove_sliced_fruits(SliceCreator.on_screen_fruits)
    return (lambda t: tuple_add(arm_loc, tuple_mul(t % 1, (LINE_LENGTH, 0)))), None, None, None


def line_trajectory(arm_loc, fruit_trajectories_and_starting_times):  # gets time in sec

    def xy_by_t(t):
        """
        trajectory in streight line
        :param t:
        :return:
        """
        T = 0.5
        x_0 = arm_loc[0]
        y_0 = arm_loc[1]
        d_a = abs(x_0 / 2.0)  # must be x_0 / 4.0 for the calculation of the acceleration
        # acc = 12.5 * abs(x_0) / T
        acc = 180.0
        t_a = math.sqrt(2 * abs(d_a / acc))
        v = acc * t_a

        x = x_0
        y = y_0
        y = t / T * d_a
        if t < t_a:
            x = x_0 + 0.5 * acc * t**2
        elif t_a < t < T - t_a:
            x = x_0 + d_a + v * (t - t_a)
        elif t > T - t_a:
            x = x_0 + d_a + v * (T - 2 * t_a) + v * (t - (T - t_a)) - 0.5 * acc * (t - (T - t_a))**2

        x_0 = -x_0
        v = -v
        d_a = -d_a
        acc = -acc

        if T < t < T + t_a:
            x = x_0 + 0.5 * acc * (t - T)**2
        elif T + t_a < t < 2 * T - t_a:
            x = x_0 + d_a + v * (t - T - t_a)
        elif t > 2 * T - t_a:
            x = x_0 + d_a + v * (T - 2 * t_a) + v * (t - T - (T - t_a)) - 0.5 * acc * (t - T - (T - t_a))**2
        return (x, y)

    return xy_by_t, None, None, None


def complex_slice(arm_loc, fruit_trajectories_and_starting_times):
    return (lambda t: tuple_add(tuple_add(arm_loc, (0, -2)),
                                tuple_mul(2, (math.cos(2*math.pi*t*10), math.sin(2*math.pi*t*10))))), None, None, None


def theta_slice(arm_loc, fruit_trajectories_and_starting_times):
    SliceCreator.on_screen_fruits = []

    return (lambda t: (R*math.cos(math.pi / 3 + 2 * math.pi * (1-t)/6),
                       R*math.sin(math.pi / 3 + 2 * math.pi * (1-t)/6) + r - d)),\
           None, None, None


def radius_slice(arm_loc, fruit_trajectories_and_starting_times):
    SliceCreator.on_screen_fruits = []
    theta_0 = math.acos(SCREEN[0]/(2*(R+r))) + 0.07

    return (lambda t: tuple_mul((R+r),
                                (math.cos(theta_0 + (math.pi - 2* theta_0) * (1 - t)),
                                math.sin(theta_0 + (math.pi - 2* theta_0) * (1 - t))  - d/(R+r)))), \
           None, None, fruit_trajectories_and_starting_times
