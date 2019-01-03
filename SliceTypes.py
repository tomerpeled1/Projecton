import math
import Algorithmics

#
LINE_LENGTH = 10

r = 9.9
R = 14.9
d = 17.9
SCREEN = [16, 12]


def tuple_add(tup1, tup2):
    """
    adding the sum element element of 2 tuples
    :param tup1: tuple of 2 doubles
    :param tup2: tuple of 2 doubles
    :return: the sum element element of the 2 tuples
    """
    return tup1[0] + tup2[0], tup1[1] + tup2[1]


def tuple_mul(scalar, tup):
    """
    return the multiplication of a tuple by a scalar
    :param scalar: double
    :param tup: tuple of 2 doubles
    :return: the multiplication of a tuple by a scalar
    """
    return scalar * tup[0], scalar * tup[1]


def x_algorithmics_to_mechanics(x_algorithmics):
    """
    :param x_algorithmics: x coordinate in the algorithmics (sliceCreator) coordinate system: (0, 0) on bottom left
    :return: x coordinate in the Mechanics and simulation coordinate system: (0, 0) on bottom middle
            (where the base of the motor is).
    """
    return x_algorithmics - SCREEN[0]/2


def slice_to_peak(arm_loc, fruit_trajectories_and_starting_times):
    """
    make straight line to the peak in a constant speed
    :param arm_loc: (x,y) of arm location
    :param fruit_trajectories_and_starting_times: function of (x,y) by t
    :return: slice, timer, t_peak, fruit_trajectories_and_starting_times
    """
    x_arm_loc, y_arm_loc = arm_loc
    if not fruit_trajectories_and_starting_times:  # if the list of the fruits is empty - returns radius_slice
        return radius_slice(arm_loc, fruit_trajectories_and_starting_times)
    # gets the first fruit of the list
    chosen_fruit = fruit_trajectories_and_starting_times[0]
    # fruits to show their trajectory and than to delete by Algorithmics.remove_sliced_fruits(chosen_fruits)
    chosen_fruits = [chosen_fruit]
    chosen_trajectory, timer = chosen_fruit
    # the coordinates of the peak
    t_peak, (x_peak, y_peak) = chosen_trajectory.calc_peak()
    # converting the int to the right5 coordinate system
    x_peak = x_algorithmics_to_mechanics(x_peak)

    def slice_trajectory(t):
        """
        returns a tuple (x, y) by t of the trajectory of the pen - the ent of the second arm
        :param t: double time between 0 and 1
        :return: the location (x, y) of the pen in cm
        """
        x_slice = x_arm_loc + (x_peak - x_arm_loc) * t * 2
        y_slice = y_arm_loc + (y_peak - y_arm_loc) * t * 2
        return x_slice, y_slice

    fruit_trajectories_and_starting_times_copy = fruit_trajectories_and_starting_times.copy()
    # delete the chosen fruit
    Algorithmics.remove_sliced_fruits(chosen_fruits)
    return slice_trajectory, timer, t_peak, fruit_trajectories_and_starting_times_copy


def line_constant_speed(arm_loc, fruit_trajectories_and_starting_times):
    """
    returns a trajectory of straight line in constant speed
    :param arm_loc: the location of the pen - tuple (x, y) in cm
    :param fruit_trajectories_and_starting_times: the trajectories of the fruits
    :return: function of (x, y) by t of the pen, None, None, None
    """
    Algorithmics.remove_sliced_fruits(Algorithmics.on_screen_fruits)

    def slice_trajectory(t):
        """
        function of (x, y) by t in cm - t between 0 and 1
        :param t: double time between 0 and 1
        :return: the location (x, y) of the pen in cm
        """
        x, y = tuple_add(arm_loc, tuple_mul(t % 1, (LINE_LENGTH, 0)))
        return x, y

    return slice_trajectory, None, None, None


def line_acceleration_trajectory(arm_loc, fruit_trajectories_and_starting_times):  # gets time in sec
    """
    returns a trajectory of straight line with acceleration in the beginning and in the end
    :param arm_loc: the location of the pen - tuple (x, y) in cm
    :param fruit_trajectories_and_starting_times: the trajectories of the fruits
    :return: function of (x, y) by t of the pen, None, None, None
    """

    def xy_by_t(t):
        """
        trajectory in streight line
        :param t: double time between 0 and 1
        :return: the location (x, y) of the pen in cm
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
    """

    :param arm_loc:
    :param fruit_trajectories_and_starting_times:
    :return:
    """
    return (lambda t: tuple_add(tuple_add(arm_loc, (0, -2)),
                                tuple_mul(2, (math.cos(2*math.pi*t*10), math.sin(2*math.pi*t*10))))), None, None, None


def theta_slice(arm_loc, fruit_trajectories_and_starting_times):
    Algorithmics.on_screen_fruits = []

    return (lambda t: (R*math.cos(math.pi / 3 + 2 * math.pi * (1-t)/6),
                       R*math.sin(math.pi / 3 + 2 * math.pi * (1-t)/6) + r - d)),\
           None, None, None


def radius_slice(arm_loc, fruit_trajectories_and_starting_times):
    Algorithmics.on_screen_fruits = []
    theta_0 = math.acos(SCREEN[0]/(2*(R+r))) + 0.07

    return (lambda t: tuple_mul((R+r),
                                (math.cos(theta_0 + (math.pi - 2* theta_0) * (1 - t)),
                                math.sin(theta_0 + (math.pi - 2* theta_0) * (1 - t))  - d/(R+r)))), \
           None, None, fruit_trajectories_and_starting_times
