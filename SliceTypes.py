def slice_to_peak(arm_loc, fruit_trajectories_and_strating_times):
    x_arm_loc, y_arm_loc = arm_loc
    chosen_trajectory = fruit_trajectories_and_strating_times[0][0]
    t_peak, (x_peak, y_peak) = chosen_trajectory.calc_peak()
    return lambda t : (t * x_peak, t * y_peak) + ((1 - t) * x_arm_loc, (1 - t) * y_arm_loc)

def stupid_slice(arm_loc, fruit_trajectories):
    return lambda t : (t * 16, 0) if t < 0.5 else ((1 - t) * 16, 0)