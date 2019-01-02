import pygame
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# plot constants.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# ---------- CONSTANTS -------------
SCREEN = (16, 12)   # dimensions of 10'' screen
ARMS = (15, 10)     # length of arm links in cm
d = 18                  # distance from screen in cm
STEPS_ROUND = 200   # steps of the motor for full round
MINIMAL_ANGLE = 2 * np.pi / STEPS_ROUND  # the angle that the motors make in full step in radian
T = 1               # time of one slice in sec
dt_serial = 0.005    # time between 2 readings from serial in sec
dt_motor = 0.0025    # time of writing to the serial in sec
times_ideal = int(T / dt_motor)  # the size of the vectors for the simulation
times_serial = int(T / dt_serial)     # the amount of different values for the


# ---------- ALGORITHMIC FUNCTION ---------------
def get_xy_by_t_line_acceleration(t):  # gets time in sec
    """
    example of trajectory function that tells the arm to make a straight line with acceleration in the beginning and
    the end of the line
    :param t: time
    :return: tuple (x, y)
    """
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
        x = x_0 + d_a + v * (T - 2 * t_a) + v * (t - (T - t_a)) - 0.5 * acc * (t - (T - t_a)) ** 2
    return x, y


def get_xy_by_t_line_const_speed(t):  # gets time in sec
    """
    example of trajectory function that tells the arm to make a straight line with constant speed
    :param t: time
    :return: tuple (x, y)
    """
    x_0 = -SCREEN[0] / 2
    y_0 = 0.5 * SCREEN[1]

    x = x_0 + SCREEN[0] * t / T
    y = y_0

    return x, y


# ----------- PLOTS AND GRAPHS FUNCTIONS -----------
def plot_screen(screen):
    """
    plots the lines for the tablet screen in the pygame simulation
    :param screen: pygame screen object - pygame.display.set_mode((WIDTH, HEIGHT))
    """
    draw_line([SCREEN[0] / 2, d], [SCREEN[0] * 3 / 2, d], screen)
    draw_line([SCREEN[0] / 2, d + SCREEN[1]], [SCREEN[0] * 3 / 2, d + SCREEN[1]], screen)
    draw_line([SCREEN[0] / 2, d], [SCREEN[0] / 2, d + SCREEN[1]], screen)
    draw_line([SCREEN[0] * 3 / 2, d], [SCREEN[0] * 3 / 2, d + SCREEN[1]], screen)
    draw_circle([SCREEN[0], 0], 2, screen)


def draw_line(start_pos, end_pos, screen):
    """
    draws a line in the pygame simulation
    :param start_pos: tuple (x, y)
    :param end_pos: tuple (x, y)
    :param screen: pygame screen object - pygame.display.set_mode((WIDTH, HEIGHT))
    """
    pygame.draw.line(screen, BLUE, [cm_to_pixels(start_pos[0]), cm_to_pixels(
        start_pos[1])], [cm_to_pixels(end_pos[0]), cm_to_pixels(end_pos[1])], 1)
    return


def draw_circle(pos, radius, screen):
    """
    draws a circle in the pygame simulation
    :param pos: tuple (x, y)
    :param radius: double
    :param screen: pygame screen object - pygame.display.set_mode((WIDTH, HEIGHT))
    """
    pygame.draw.circle(screen, RED, [cm_to_pixels(pos[0]), cm_to_pixels(pos[1])],
                       radius, 1)


def cm_to_pixels(length):
    """
    returns the length in number of pixels
    :param length: double in cm
    :return: int - number of pixels
    """
    return int(20 * length)


def draw_graph(x, y, title, xlabel, ylabel):
    """
    draws a graph in matplotlib
    :param x: vector for x axis
    :param y: vector for y axis
    :param title: string for title
    :param xlabel: string for x axis lable
    :param ylabel: string for y axis lable
    """
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


WIDTH = cm_to_pixels(2 * SCREEN[0])
HEIGHT = cm_to_pixels(2 * (SCREEN[1] + d))


# ------------- CALCULATION FUNCTIONS ------------
def modulo_by_1(num):
    """
    makes a modulo by 1 that returns a positive number for positive parameter and a negative number for a negative
    parameter
    :param num: double
    """
    if num > 0:
        return num % 1
    else:
        return num % 1 - 1


def get_angles_by_xy_and_dt(get_xy_by_t, dt):
    """
    returns theta and phi in intervals of dt by the function "get_xy_by_t"
    :param get_xy_by_t: a function that gets a double t between 0 and 1 and returns tuple (x, y)
    :param dt: interval to make theta and phi - double
    :return: tuple (vector of theta, vector of phi)
    """
    times = range(int(T / dt))
    # get xy by dt
    xy = [[0 for _ in times], [0 for _ in times]]
    for i in times:
        xy[0][i], xy[1][i] = get_xy_by_t(dt * i)

    # calc angles by xy
    r = np.sqrt(np.power(xy[0], 2) + np.power(np.add(d, xy[1]), 2))
    alpha = np.arctan2(np.add(d, xy[1]), xy[0])  # angle
    # between r and x axis
    a = np.add(-1, np.remainder(np.add(1, np.multiply(np.add(-math.pow(ARMS[0], 2) - math.pow(ARMS[1], 2),
                                                             np.power(r, 2)), 1.0 / (2 * ARMS[0] * ARMS[1]))), 2))
    beta = np.arccos(a)
    b = np.add(-1, np.remainder(np.add(1, np.multiply(np.add(math.pow(ARMS[0], 2) - math.pow(ARMS[1], 2),
                                                             np.power(r, 2)), 1.0 / (2 * ARMS[0] * r))), 2))
    delta = np.arccos(b)
    # angle between r and 1st link
    theta = alpha + delta
    phi = theta - beta

    # draw_graph(times, theta, "theta to time", "time", "theta")
    # draw_graph(times, phi, "phi to time", "time", "phi")

    return theta, phi


def make_slice_by_trajectory(get_xy_by_t):
    """
    makes a slice in the pygame simulation
    :param get_xy_by_t: function that gets t (double between 0 and 1) and returns a tuple (x, y)
    :return: tuple (theta vector, phi vector)
    """
    # get the vectors of theta and phi
    theta_vector, phi_vector = get_angles_by_xy_and_dt(get_xy_by_t, dt_serial)

    # calculate the steps for each motor
    steps_theta, steps_phi = calc_steps_theta_and_steps_phi_by_theta_and_phi(theta_vector, phi_vector)

    # print("steps theta: ")
    # print(steps_theta)
    # print("steps phi: ")
    # print(steps_phi)

    # get the theta vector and phi vector to show in simulation
    theta_simulation, phi_simulation = duplicate_theta_and_phi_values_for_simulation(theta_vector, phi_vector,
                                                                                     steps_theta, steps_phi)

    return theta_simulation, phi_simulation


def calc_steps_theta_and_steps_phi_by_theta_and_phi(theta_vector, phi_vector):
    """
    calculate the steps for each motor by theta and phi vectors
    :param theta_vector: list of theta angles in dt interval
    :param phi_vector: list of phi angles in dt interval
    :return: tuple of 2 lists (steps for theta motor, steps for phi motor)
    """
    # get the subtractions of theta and phi
    d_theta, d_phi = np.diff(theta_vector), np.diff(phi_vector)
    # convert to steps units
    steps_theta_decimal, steps_phi_decimal = ((1 / MINIMAL_ANGLE) * d_theta), ((1 / MINIMAL_ANGLE) * d_phi)
    # improve accuracy by adding the modulo 1 of the previous steps
    for i in range(times_serial-2):
        steps_theta_decimal[i+1] += modulo_by_1(steps_theta_decimal[i])
        steps_phi_decimal[i+1] += modulo_by_1(steps_phi_decimal[i])
    # cast to int type
    steps_theta = steps_theta_decimal.astype(int)
    steps_phi = steps_phi_decimal.astype(int)
    return steps_theta, steps_phi


def duplicate_theta_and_phi_values_for_simulation(theta_vector, phi_vector, steps_theta, steps_phi):
    # the vectors for running the simulation - in the ideal dt
    theta_simulation = [0 for _ in range(times_ideal)]
    phi_simulation = [0 for _ in range(times_ideal)]

    # initialize the first angles
    theta_simulation[0] = theta_vector[0]
    phi_simulation[0] = phi_vector[0]

    # make the delay between 2 moves - the delay is according to the time left to fill the dt_serial
    angle_move_index = 0
    times_ratio = int(times_ideal / times_serial)
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


# real solution


# ------------- CALCULATE LOCATIONS -------------
def run_simulation(func, fruits_trajectories):

    print(fruits_trajectories)

    # the ideal angles like in the function of the algorithmic
    theta_ideal, phi_ideal = make_ideal_slice_by_trajectory(func)

    # the practical angles
    theta_practical, phi_practical = make_slice_by_trajectory(func)

    # ------------- PLOT -------------------
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    x_0, y_0 = SCREEN[0], 0

    errors = [0 for _ in range(times_ideal)]
    x_ideal_vector = [0 for _ in range(times_ideal)]
    y_ideal_vector = [0 for _ in range(times_ideal)]
    x_practical_vector = [0 for _ in range(times_ideal)]
    y_practical_vector = [0 for _ in range(times_ideal)]
    time_vector = [dt_motor * i for i in range(times_ideal)]  # TODO why is it here? delete if isn't used.

    # loop of plot
    for i in range(times_ideal):
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            running = 0  # TODO why is it here? delete of isn't used.

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

        # draw fruits locations


        pygame.display.flip()

        time.sleep(dt_motor)
    time.sleep(2)
    pygame.display.quit()
    # draw_graph(x_ideal_vector, y_ideal_vector, "ideal", "x", "y")
    # draw_graph(x_practical_vector, y_practical_vector, "practical", "x", "y")

    # plots error graph
    # draw_graph(time_vector, errors, "errors to time", "time", "error")
