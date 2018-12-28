import math
import SliceCreator

LINE_LENGTH = 20


def tuple_add(tup1, tup2):
    return tup1[0] + tup2[0], tup1[1] + tup2[1]


def tuple_mul(scalar, tup):
    return scalar * tup[0], scalar * tup[1]


def slice_to_peak(arm_loc, fruit_trajectories_and_starting_times):
    x_arm_loc, y_arm_loc = arm_loc
    chosen_fruit = fruit_trajectories_and_starting_times[0]
    chosen_fruits = [chosen_fruit]
    chosen_trajectory, timer = chosen_fruit
    t_peak, (x_peak, y_peak) = chosen_trajectory.calc_peak()
    slice = lambda t: (t * x_peak, t * y_peak) + ((1 - t) * x_arm_loc, (1 - t) * y_arm_loc)
    SliceCreator.remove_sliced_fruits(chosen_fruits)
    return slice, timer, t_peak


def stupid_slice(arm_loc, fruit_trajectories):
    SliceCreator.remove_sliced_fruits(SliceCreator.on_screen_fruits)
    return (lambda t: tuple_add(arm_loc, tuple_mul(t % 1, (2 * LINE_LENGTH,
            0)) if (t % 1) < 0.5 else tuple_mul((1-t) % 1, (2 * LINE_LENGTH,
            0)))), None, None


def complex_slice(arm_loc, fruit_trajectories):
    return (lambda t: tuple_add(tuple_add(arm_loc, (0, -2)),
                                tuple_mul(2, (math.cos(2*math.pi*t*10), math.sin(2*math.pi*t*10))))), None, None
