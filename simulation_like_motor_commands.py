import pygame
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import SliceCreator


#plot constants.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# ---------- ALGORITHMIC FUNCTION ---------------
def get_xy_by_t(t):  # gets time in sec
    acc = 1800.0
    x_0 = -SCREEN[0] / 2
    y_0 = 0.8 * SCREEN[1]
    d_a = SCREEN[0] / 4.0
    t_a = math.sqrt(2 * d_a / acc)
    v = acc * t_a

    x = x_0
    y = y_0

    if t < t_a:
        x = x_0 + 0.5 * acc * math.pow(t, 2)
    elif t_a < t < T - t_a:
        x = x_0 + d_a + v * (t - t_a)
    elif t > T - t_a:
        x = x_0 + d_a + v * (T - 2 * t_a) - 0.5 * acc * math.pow(t - (T -
                                                                      t_a), 2)
    return x, y



def get_xy_by_t_simple(t):  # gets time in sec
    x_0 = -SCREEN[0] / 2
    y_0 = 0.5 * SCREEN[1]

    x = x_0 + SCREEN[0] * t / T
    y = y_0

    return x, y

def algorithmic_parametrization():
    return SliceCreator.create_slice()

# ----------- PLOTS AND GRAPHS FUNCTIONS -----------
def plot_screen(screen):
    draw_line([SCREEN[0] / 2, d], [SCREEN[0] * 3 / 2, d], screen)
    draw_line([SCREEN[0] / 2, d + SCREEN[1]], [SCREEN[0] * 3 / 2, d + SCREEN[1]], screen)
    draw_line([SCREEN[0] / 2, d], [SCREEN[0] / 2, d + SCREEN[1]], screen)
    draw_line([SCREEN[0] * 3 / 2, d], [SCREEN[0] * 3 / 2, d + SCREEN[1]], screen)
    draw_circle([SCREEN[0], 0], 2, screen)


def draw_line(start_pos, end_pos, screen):
    pygame.draw.line(screen, BLUE, [to_pixels(start_pos[0]), to_pixels(
        start_pos[1])], [to_pixels(end_pos[0]), to_pixels(end_pos[1])], 1)
    return


def draw_circle(pos, radius, screen):
    pygame.draw.circle(screen, RED, [to_pixels(pos[0]), to_pixels(pos[1])],
                       radius, 1)


def to_pixels(length):
    return int(20 * length)


def draw_graph(x, y, title, xlabel, ylabel):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# ------------- CALCULATION FUNCTIONS ------------
def modulo(a, n):
    if a > 0:
        return a%n
    else:
        return a%n - 1


def get_angles_by_xy_and_dt(get_xy_by_t, dt):

    time = int(T / dt)
    times = range(time)
    # get xy by dt
    xy = [[0 for i in times], [0 for i in times]]
    for i in times:
        xy[0][i], xy[1][i] = get_xy_by_t(dt * i)

    # calc angles by xy
    r = np.sqrt(np.power(xy[0], 2) + np.power(np.add(d, xy[1]), 2))
    alpha = np.arctan2(np.add(d, xy[1]), xy[0])  # angle
    # between r and x axis
    a = np.add(-1, np.remainder(np.add(1, np.multiply(np.add(-math.pow(ARMS[
                                                                           0], 2) - math.pow(
        ARMS[1], 2), np.power(r, 2)), 1.0 / (2 * ARMS[0] * ARMS[1]))), 2))
    beta = np.arccos(a)
    b = np.add(-1, np.remainder(np.add(1, np.multiply(np.add(math.pow(ARMS[0],
                                                            2) - math.pow(ARMS
            [1], 2), np.power(r, 2)), 1.0 / (2 * ARMS[0] * r))), 2))
    delta = np.arccos(b)
    # angle between r and 1st link
    theta = alpha + delta
    phi = theta - beta

    # draw_graph(times, theta, "theta to time", "time", "theta")
    # draw_graph(times, phi, "phi to time", "time", "phi")

    return theta, phi


def make_slice_by_trajectory(get_xy_by_t):
    theta, phi = get_angles_by_xy_and_dt(get_xy_by_t, dt_serial)
    d_theta, d_phi = np.diff(theta), np.diff(phi)
    steps_theta_decimal, steps_phi_decimal = ((1 / MINIMAL_ANGLE) * d_theta), \
                             ((1 / MINIMAL_ANGLE) * d_phi)
    for i in range(times_serial-2):
        steps_theta_decimal[i+1] += modulo(steps_theta_decimal[i], 1)
        steps_phi_decimal[i+1] += modulo(steps_phi_decimal[i], 1)
    steps_theta = steps_theta_decimal.astype(int)
    steps_phi = steps_phi_decimal.astype(int)

    print(steps_theta)
    print('*********************')
    print(steps_phi)
    # steps_theta, steps_phi = ((1 / MINIMAL_ANGLE) * d_theta).astype(int), \
    #                          ((1 / MINIMAL_ANGLE) * d_phi).astype(int)

    # the vectors for running the simulation - in the ideal dt
    theta_simulation = [0 for i in range(times_ideal)]
    phi_simulation = [0 for i in range(times_ideal)]

    # initialize the first angles
    theta_simulation[0] = theta[0]
    phi_simulation[0] = phi[0]

    angle_move_index = 0
    times_ratio = (int)(times_ideal / times_serial)
    for i in range(times_ideal-1):
        if i % times_ratio == 0 and angle_move_index < len(steps_theta):
            if abs(steps_theta[angle_move_index]) == 0:
                theta_simulation[i + 1] = theta_simulation[i]
            else:
                for j in range(abs(steps_theta[angle_move_index])):
                    theta_simulation[i + j + 1] = theta_simulation[i + j] + \
                                np.sign(steps_theta[angle_move_index]) * MINIMAL_ANGLE
            if abs(steps_phi[angle_move_index]) == 0:
                phi_simulation[i + 1] = phi_simulation[i]
            else:
                for j in range(abs(steps_phi[angle_move_index])):
                    phi_simulation[i + j + 1] = phi_simulation[i + j] + \
                                np.sign(steps_phi[angle_move_index]) * MINIMAL_ANGLE
            angle_move_index += 1
        else:
            if theta_simulation[i + 1] == 0:
                theta_simulation[i + 1] = theta_simulation[i]
            if phi_simulation[i + 1] == 0:
                phi_simulation[i + 1] = phi_simulation[i]
    return theta_simulation, phi_simulation


def make_ideal_slice_by_trajectory(get_xy_by_t):
    theta, phi = get_angles_by_xy_and_dt(get_xy_by_t, dt_motor)
    return theta, phi

def xy_by_theta_phi(theta, phi, x_0):
    x = x_0 + ARMS[0] * np.cos(theta) + ARMS[1] * np.cos(phi)
    y = ARMS[0] * np.sin(theta) + ARMS[1] * np.sin(phi)
    return x, y


def xy_by_theta(theta, x_0):
    x = x_0 + ARMS[0] * np.cos(theta)
    y = ARMS[0] * np.sin(theta)
    return x, y

# ---------- CONSTANTS -------------
SCREEN = [16, 12]   # dimensions of 10'' screen
ARMS = [15, 10]     # length of arm links in cm
# density = 7         # gr/cm
# link_mass = 20      # mass of link in gr
# pen_mass = 10       # mass of end in gr
d = 10               # distance from screen in cm
# MOTOR_SPEED = 50    # angular speed of motor in rpm
STEPS_ROUND = 200   # steps of the motor for full round
MINIMAL_ANGLE = 2 * np.pi / STEPS_ROUND
T = 1               # time of one slice in sec
dt_serial = 0.005    # time between 2 readings from serial in sec
dt_motor = 0.0025    # time of writing to the serial in sec
times_ideal = int(T / dt_motor)  # the size of the vectors for the simulation
times_serial = int(T / dt_serial)     # the amount of different values for the
# real solution

# ------------- CALCULATE LOCATIONS -------------
def run_simulation(func, SCREEN = SCREEN):
    # the ideal angles like in the function of the algorithmic
    theta_ideal, phi_ideal = make_ideal_slice_by_trajectory(algorithmic_parametrization())

    # the practical angles
    theta_practical, phi_practical = make_slice_by_trajectory(algorithmic_parametrization())

    # ------------- PLOT -------------------

    WIDTH = to_pixels(2 * SCREEN[0])
    HEIGHT = to_pixels(2 * (SCREEN[1] + d))

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    x_0, y_0 = SCREEN[0], 0

    errors = [0 for i in range(times_ideal)]
    x_ideal_vector = [0 for i in range(times_ideal)]
    y_ideal_vector = [0 for i in range(times_ideal)]
    x_practical_vector = [0 for i in range(times_ideal)]
    y_practical_vector = [0 for i in range(times_ideal)]
    time_vector = [dt_motor * i for i in range(times_ideal)]

    # loop of plot
    for i in range(times_ideal):
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            running = 0

        screen.fill(WHITE)
        plot_screen(screen)
        # ideal locations
        x_ideal, y_ideal = xy_by_theta_phi(theta_ideal[i], phi_ideal[i], x_0)
        x_ideal_vector[i], y_ideal_vector[i] = x_ideal, y_ideal
        x_link_ideal, y_link_ideal = xy_by_theta(theta_ideal[i], x_0)
        draw_circle([x_ideal, y_ideal], 2, screen)
        draw_line([x_link_ideal, y_link_ideal], [x_0, y_0], screen)
        draw_line([x_link_ideal, y_link_ideal], [x_ideal, y_ideal], screen)

        # real locations
        x_practical, y_practical = xy_by_theta_phi(theta_practical[i],
                                                   phi_practical[i], x_0)
        x_practical_vector[i], y_practical_vector[i] = x_practical, y_practical
        x_link_practical, y_link_practical = xy_by_theta(theta_practical[i], x_0)
        draw_circle([x_practical, y_practical], 2, screen)
        draw_line([x_link_practical, y_link_practical], [x_0, y_0], screen)
        draw_line([x_link_practical, y_link_practical], [x_practical, y_practical], screen)

        errors[i] = math.sqrt(math.pow(x_practical - x_ideal, 2) + math.pow(y_practical -
                                                                            y_ideal, 2))

        pygame.display.flip()

        time.sleep(dt_motor)

    # draw_graph(x_ideal_vector, y_ideal_vector, "ideal", "x", "y")
    # draw_graph(x_practical_vector, y_practical_vector, "practical", "x", "y")

    # plots error graph
    # draw_graph(time_vector, errors, "errors to time", "time", "error")

if __name__ == '__main__':
    run_simulation(algorithmic_parametrization())