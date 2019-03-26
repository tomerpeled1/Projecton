"""
this file creates slice routs - (x,y)(t).
"""

import math

LINE_LENGTH = 10

r = 10  # second arm length in cm
R = 15  # first arm length in cm
d = 15  # distance of major axis from screen in cm
SCREEN = [16, 12]  # (x,y) dimensions of screen in cm
SLICE_ZONE = 0.5
PARTITION = 20
MINIMAL_DISTANCE_FOR_PARTITION = 1
LINEAR = 0
ANGLES = 1


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


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
    return x_algorithmics - SCREEN[0] / 2


def slice_to_peak(arm_loc, fruit_trajectories_and_starting_times):  # TODO update to new algo
    """
    make straight line to the peak in a constant speed
    :param arm_loc: (x,y) of arm location
    :param fruit_trajectories_and_starting_times: list of
            [function of (x,y) by t, and double starting time]
    :return: slice, timer, t_peak, fruit_trajectories_and_starting_times
    """
    if not fruit_trajectories_and_starting_times:  # if the list of the fruits is empty - returns radius_slice
        return linear_slice(arm_loc, fruit_trajectories_and_starting_times)
    # gets the first fruit of the list
    chosen_fruit = fruit_trajectories_and_starting_times[0]
    # fruits to show their trajectory and than to delete by Algorithmics.remove_sliced_fruits(chosen_fruits)
    chosen_fruits = [chosen_fruit]
    chosen_trajectory, time_created = chosen_fruit
    # the coordinates of the peak
    t_peak, (x_peak, y_peak) = chosen_trajectory.calc_peak()
    if distance((SCREEN[0] / 2, -d), (x_peak, y_peak)) > R + r:
        return linear_slice(arm_loc, fruit_trajectories_and_starting_times)
    if not in_bound((x_peak, y_peak)):
        return linear_slice(arm_loc, fruit_trajectories_and_starting_times)
    # converting the int to the right coordinate system
    x_peak = x_algorithmics_to_mechanics(x_peak)

    # TODO add 1.1 factor to x_peak,y_peak

    slice_trajectory = linear_slice_between_2_points(arm_loc, (x_peak, y_peak))

    fruit_trajectories_and_starting_times_copy = fruit_trajectories_and_starting_times.copy()
    # delete the chosen fruit
    # TODO sleep until slice (still TODO?)

    return slice_trajectory, time_created, t_peak, fruit_trajectories_and_starting_times_copy


def linear_slice_between_2_points(start, target):
    """
    Creates a slice in straight line to the given target
    :param start: the location of the arm at beginning of slice.
    :param target: the wanted location of the arm at end of slice.
    :return: function of (x,y) as function of t
    """
    x_start, y_start = start
    x_target, y_target = target

    def slice_trajectory(t):
        """
        returns a tuple (x, y) by t of the trajectory of the pen - the end of the second arm
        :param t: double time between 0 and 1
        :return: the location (x, y) of the pen in cm
        """
        x_slice = x_start + (x_target - x_start) * t
        y_slice = y_start + (y_target - y_start) * t
        return x_slice, y_slice

    return slice_trajectory


def line_acceleration_trajectory(arm_loc, _):  # gets time in sec
    """
    returns a trajectory of straight line with acceleration in the beginning and in the end
    :param arm_loc: the location of the pen - tuple (x, y) in cm
    :param _: the trajectories of the fruits
    :return: function of (x, y) by t of the pen, None, None, None
    """

    def xy_by_t(t):
        """
        trajectory in straight line
        :param t: double time between 0 and 1
        :return: the location (x, y) of the pen in cm
        """
        t_tot = 0.5
        x_0 = arm_loc[0]
        # y_0 = arm_loc[1]
        d_a = abs(x_0 / 2.0)  # must be x_0 / 4.0 for the calculation of the acceleration
        # acc = 12.5 * abs(x_0) / t_tot
        acc = 180.0
        t_a = math.sqrt(2 * abs(d_a / acc))
        v = acc * t_a

        x = x_0
        y = t / t_tot * d_a
        if t < t_a:
            x = x_0 + 0.5 * acc * t ** 2
        elif t_a < t < t_tot - t_a:
            x = x_0 + d_a + v * (t - t_a)
        elif t > t_tot - t_a:
            x = x_0 + d_a + v * (t_tot - 2 * t_a) + v * (t - (t_tot - t_a)) - 0.5 * acc * (t - (t_tot - t_a)) ** 2

        x_0 = -x_0
        v = -v
        d_a = -d_a
        acc = -acc

        if t_tot < t < t_tot + t_a:
            x = x_0 + 0.5 * acc * (t - t_tot) ** 2
        elif t_tot + t_a < t < 2 * t_tot - t_a:
            x = x_0 + d_a + v * (t - t_tot - t_a)
        elif t > 2 * t_tot - t_a:
            x = x_0 + d_a + \
                v * (t_tot - 2 * t_a) + v * (t - t_tot - (t_tot - t_a)) - 0.5 * acc * (t - t_tot - (t_tot - t_a)) ** 2
        return x, y

    return xy_by_t, None, None, None


# def complex_slice(arm_loc, _):
#     """
#
#     :param arm_loc: the location of the pen - tuple (x, y) in cm.
#     :param _:the trajectories of the fruits
#     :return: function of (x, y) by t of the pen, None, None, None
#     """
#
#     def ret_slice(t):
#         location = (math.cos(2 * math.pi * t * 10), math.sin(2 * math.pi * t * 10))
#         location = tuple_mul(2, location)
#         new_arm_loc = tuple_add(arm_loc, (0, -2))
#         return tuple_add(location, new_arm_loc)
#
#     # ---- this is the old lambda, for cases the new code does not work.-------#
#
#     # (lambda t: tuple_add(tuple_add(arm_loc, (0, -2)),
#     #                         tuple_mul(2, (
#     #                         math.cos(2 * math.pi * t * 10), math.sin(2 * math.pi * t * 10))))), None, None, None
#
#     return ret_slice, None, None, None


def slice_through_many_points(arm_loc, ordered_points, move_between_points=LINEAR):
    """

    :param arm_loc:
    :param ordered_points: list of tuples, points in (x,y) at mechanics coordinates.
    :param move_between_points:
    :return:
    """
    if move_between_points == LINEAR:
        move_func = linear_slice_between_2_points
    else:
        move_func = linear_slice_between_2_points
    slice_parts = [move_func(arm_loc, ordered_points[0])]
    for i in range(len(ordered_points)-1):
        slice_parts.append(move_func(ordered_points[i], ordered_points[i+1]))
    # print ("*&*&*&*&* SLICE_PARTS before unite ", slice_parts)
    slice_to_return = unite_slice(slice_parts)
    xy_points = get_partition(slice_to_return)
    return xy_points


def unite_slice(slice_parts):
    # print ("*&*&*&*&* SLICE_PARTS ", slice_parts)
    n = len(slice_parts)
    time_for_part = 1.0/n

    def united_slice(t):
        if t != 1:
            i = int(t/time_for_part)
            relative_time = t - i*time_for_part
            return slice_parts[i](relative_time*n)
        else:
            return slice_parts[-1](1)
    return united_slice


def radius_slice(_, __):
    """
    the slice that uses only theta - stupid and simple.
    :return: function of (x, y) by t of the pen, None, None, None
    """
    theta_0 = math.acos(SCREEN[0] / (2 * (R + r))) + 0.07

    def ret_slice(t):
        # not normalized x and y
        x_loc = math.cos(theta_0 + (math.pi - 2 * theta_0) * t)
        y_loc = math.sin(theta_0 + (math.pi - 2 * theta_0) * t) - d / (R + r)

        return tuple_mul(r + R, (x_loc, y_loc))

    xy_points = get_partition(ret_slice)

    return xy_points


def in_bound(point, percent=SLICE_ZONE):
    left_bound = 0
    right_bound = SCREEN[0]
    upper_bound = (1.0 - percent) * SCREEN[1]
    lower_bound = 0
    x, y = point
    if left_bound < x < right_bound and lower_bound < y < upper_bound:
        return True
    # print(x, y)
    return False


def get_partition(xy_by_t):
    dt = 1.0/PARTITION
    x_points = [xy_by_t(0)[0]]
    y_points = [xy_by_t(0)[1]]
    for i in range(1, PARTITION):
        current_x, current_y = xy_by_t(i*dt)
        if distance((x_points[-1], y_points[-1]), (current_x, current_y)) >= 1:
            x_points.append(current_x)
            y_points.append(current_y)
    x_final, y_final = xy_by_t(1)
    x_points.append(x_final)
    y_points.append(y_final)
    return [(x_points[i], y_points[i]) for i in range(len(x_points))]


def linear_slice(arm_loc, _):
    """
    Does a linear slice of 10 cm length.
    :param arm_loc: current location.
    :param _:
    :return: slice.
    """
    x_arm_loc = arm_loc[0]
    y_arm_loc = arm_loc[1]
    x_final = x_arm_loc - 7.0

    def xy_by_t(t):
        x_slice = x_arm_loc + (x_final - x_arm_loc) * t * 2
        y_slice = y_arm_loc
        return x_slice, y_slice

    xy_points = get_partition(xy_by_t)
    return xy_points
