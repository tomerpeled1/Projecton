import math
from scipy.optimize import curve_fit

RELATIVE_ACC = 1
ARM_DELAY = 1
CROP_SIZE = (200, 600)
FRAME_SIZE = (720, 1080)
SCREEN_SIZE = (26, 18)

class Trajectory:
    def __init__(self, x0, v, theta):
        self.x0 = x0
        self.v = v
        self.theta = theta

    def calc_trajectory(self):
        return lambda t: (x_trajectory(t, self.x0, self.v, self.theta), y_trajectory(t, self.v, self.theta))

    def calc_peak(self):
        t = self.v * math.sin(self.theta) / RELATIVE_ACC
        return t,  self.calc_trajectory()(t)

def create_slice(fruit_pix_locs):
    fruits_locs = [[pixel2cm(pix_loc) for pix_loc in fruit] for fruit in fruit_pix_locs]
    fruit_trajectories = [get_trajectory(fruit_locs) for fruit_locs in fruits_locs]
    slice_path = calc_slice(fruit_trajectories)
    return slice_path

def pixel2cm(pix_loc):
    (i_coord_crop, j_coord_crop) = pix_loc
    i_coord_frame = i_coord_crop
    j_coord_frame = FRAME_SIZE[1] / 2 - CROP_SIZE[1] / 2 + j_coord_crop
    i_coord_screen =  i_coord_frame / FRAME_SIZE[0] * SCREEN_SIZE[0]
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
    return (x - x0) * math.tan(theta) - RELATIVE_ACC * (x - x0) ** 2 / (2 * v ** 2 * math.cos(theta) ** 2)

def x_trajectory(t, x0, v, theta):
    return x0 + v * math.cos(theta) * t

def y_trajectory(t, v, theta):
    return v * math.sin(theta) * t - 0.5 * RELATIVE_ACC * t ** 2

def calc_slice(fruit_trajectories):
    x_arm_loc, y_arm_loc = get_arm_loc()
    chosen_trajectory = fruit_trajectories[0]
    t_peak, (x_peak, y_peak) = chosen_trajectory.calc_peak()
    return lambda t : (t * x_peak, t * y_peak) + ((1 - t) * x_arm_loc, (1 - t) * y_arm_loc)

def get_arm_loc():
    return 0, 90

def init_info(crop_size, frame_size, screen_size):
    global CROP_SIZE, FRAME_SIZE, SCREEN_SIZE
    CROP_SIZE = crop_size
    FRAME_SIZE = frame_size
    SCREEN_SIZE = screen_size

if __name__ == "__main__":
    pass