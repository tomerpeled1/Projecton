import math
def tuple_add(tup1, tup2):
    return (tup1[0] + tup2[0], tup1[1] + tup2[1])

def tuple_mul(scalar, tup):
    return (scalar * tup[0], scalar * tup[1])
def slice_to_peak(arm_loc, fruit_trajectories_and_strating_times):
    x_arm_loc, y_arm_loc = arm_loc
    chosen_trajectory = fruit_trajectories_and_strating_times[0][0]
    t_peak, (x_peak, y_peak) = chosen_trajectory.calc_peak()
    return lambda t : (t * x_peak, t * y_peak) + ((1 - t) * x_arm_loc, (1 - t) * y_arm_loc)

def stupid_slice(arm_loc, fruit_trajectories):
    return lambda t : tuple_add(arm_loc, tuple_mul(t, (21, 0)) if t < 0.5 else tuple_mul(1 - t, (21, 0)))

def complex_slice(arm_loc, fruit_trajectories):
    return lambda t : tuple_add(tuple_add(arm_loc, (0, -2)), tuple_mul(2,(math.cos(2*math.pi*t*10), math.sin(2*math.pi*t*10))))