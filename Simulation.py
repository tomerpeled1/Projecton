"""
the simulation, simulates the motors and work in the same coordinate system.
"""
import pygame
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import ArduinoCommunication as Ac
import Algorithmics as Algo


# ---------- CONSTANTS -------------
# COLORS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# MECHANICS
SCREEN = (16, 12)  # dimensions of 10'' screen
ARMS = (15, 10)  # length of arm links in cm
d = 15  # distance from screen in cm
STEPS_PER_REVOLUTION = 200  # number of full steps to make a full round
STEPS_FRACTION = 8  # resolution of micro-stepping
MINIMAL_ANGLE = 2 * np.pi / (STEPS_PER_REVOLUTION * STEPS_FRACTION)  # the minimal angle step of the motor in rad (it is
BITS_PER_BYTE = 8
LENGTH_OF_COMMAND = 6  # how many chars are in one command

# TIME
WANTED_RPS = 0.5  # revolutions per second of motors
ONE_STEP_DELAY = 5.0 / WANTED_RPS / STEPS_FRACTION / 1000.0  # in sec
SERIAL_BPS = 19200  # bits per second the serial can read and write
WRITE_DELAY = 1.0/(SERIAL_BPS/BITS_PER_BYTE/LENGTH_OF_COMMAND)  # delay in sec after writing to prevent buffer overload
T = 1  # time of one slice in sec
dt_serial = WRITE_DELAY * 4  # time between 2 readings from serial in sec
dt_motor = ONE_STEP_DELAY * 4  # time of writing to the serial in sec
times_ideal = int(T / dt_motor)  # the size of the vectors for the simulation
times_serial = int(T / dt_serial)  # the amount of different values for the
TIME_TO_QUIT_SIMULATION = 2  # time to quit the simulation after finished in sec

first_point = 0, 0  # theta, phi - will be updated in the code
theta_simulation, phi_simulation = [], []

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
    draw_circle([SCREEN[0], 0], 2, screen, RED)


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


def draw_circle(pos, radius, screen, color):
    """
    draws a circle in the pygame simulation
    :param pos: tuple (x, y)
    :param radius: double
    :param screen: pygame screen object - pygame.display.set_mode((WIDTH, HEIGHT))
    :param color: color of circle
    """
    pygame.draw.circle(screen, color, [cm_to_pixels(pos[0]), cm_to_pixels(pos[1])],
                       radius, 0)


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


# width and height of the window of the pygame simulation
WIDTH = cm_to_pixels(2 * SCREEN[0])
HEIGHT = cm_to_pixels(2 * (SCREEN[1] + d))


# ------------- HELPING FUNCTIONS ------------
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


def xy_by_theta_phi(theta, phi):
    """
    returns the location in (x, y) according to the location in (theta, phi) by the trigonometric connection - this
    is the location of the pen (the end of the 2nd arm)
    :param theta: double
    :param phi: double
    :param x_0: double
    :return: tuple of doubles (x, y)
    """
    x = ARMS[0] * np.cos(theta) + ARMS[1] * np.cos(phi)
    y = ARMS[0] * np.sin(theta) + ARMS[1] * np.sin(phi) - d
    return x, y


def xy_by_theta(theta, x_0):
    """
    returns the location in (x, y) according to the location in (theta) by the trigonometric connection - this
    is the location of the link (the end of the 1st arm)
    :param theta: double
    :param x_0: double
    :return: tuple of doubles (x, y)
    """
    x = x_0 + ARMS[0] * np.cos(theta)
    y = ARMS[0] * np.sin(theta)
    return x, y


def unite_vector(a):
    united = []
    for i in a:
        united += i
    return united


def draw_points(points_to_draw):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    x_0, y_0 = SCREEN[0], 0
    green = 0

    # quiting option
    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        pygame.quit()

    # plotting the screen
    screen.fill(WHITE)
    plot_screen(screen)

    for point in points_to_draw:
        x_point = point[0]
        y_point = point[1]

        x_point_simulation, y_point_simulation = xy_board_to_xy_simulation(x_point, y_point)
        color = (255, green, 0)
        draw_circle([x_point_simulation, y_point_simulation], 2, screen, color)
        # change the color to be more blue
        green += 255 // len(points_to_draw)

    # display the simulation
    pygame.display.flip()

    # quiting the simulation
    time.sleep(TIME_TO_QUIT_SIMULATION)
    pygame.display.quit()


def xy_board_to_xy_simulation(x_board, y_board):
    return x_board + SCREEN[0], y_board + d


# -------------- SLICE TRAJECTORY ------------------
def make_slice_by_trajectory(points, invert=True):
    """
    Sends commands to Arduino according to the given route from the algorithmic module.
    :param points: list of tuples, each tuple is a point the arm should go through
    :param invert: if true then make also invert slice
    """
    steps_theta, steps_phi = list(), list()
    for i in range(len(points)-1):
        current_point = Ac.xy2angles(points[i])  # in (theta,phi)
        next_point = Ac.xy2angles(points[i+1])  # in (theta,phi)
        current_steps_theta, current_steps_phi = Ac.generate_steps_list(Ac.rad2steps(next_point[0] - current_point[0]),
                                                                     Ac.rad2steps(next_point[1] - current_point[1]))
        for j in range(len(current_steps_theta)):
            steps_theta.append(current_steps_theta[j])
            steps_phi.append(current_steps_phi[j])
    global first_point
    if len(points) != 0:
        first_point = Ac.xy2angles(points[0])
    move_2_motors_simulation(steps_theta, steps_phi)
    if invert:
        i_steps_theta, i_steps_phi = Ac.invert_slice(steps_theta, steps_phi)
        if len(points) != 0:
            first_point = Ac.xy2angles(points[-1])

        move_2_motors_simulation(i_steps_theta, i_steps_phi, True)


def duplicate_theta_and_phi_values_for_simulation(theta_vector, phi_vector):
    """
    returns theta vector and phi vector to show in simulation - with the dt of the simulation
    :param theta_vector: list of theta angles - double list
    :param phi_vector: list of phi angles - double list
    :param steps_theta: steps for the theta motor - int list
    :param steps_phi: steps for the phi motor - int list
    :return: tuple of 2 lists (theta vector, phi vector) in the right dt interval
    """
    factor = int(times_ideal/times_serial)
    theta_mul = [[angle]*factor for angle in theta_vector]
    phi_mul = [[angle]*factor for angle in phi_vector]
    theta_simulation = unite_vector(theta_mul)
    phi_simulation = unite_vector(phi_mul)

    return theta_simulation, phi_simulation


def get_theta_and_phi_vectors_by_steps_vectors(steps_theta, steps_phi):
    theta_vector = []
    phi_vector = []
    theta_vector.append(first_point[0])
    phi_vector.append(first_point[1])
    for i in range(len(steps_theta)):
        theta_vector.append(theta_vector[-1] + steps_theta[i] * MINIMAL_ANGLE)
        phi_vector.append(phi_vector[-1] + steps_phi[i] * MINIMAL_ANGLE)
    return theta_vector, phi_vector


def move_2_motors_simulation(steps_theta, steps_phi, inverse=False):
    global theta_simulation, phi_simulation
    theta_vector, phi_vector = get_theta_and_phi_vectors_by_steps_vectors(steps_theta, steps_phi)
    theta_simulation, phi_simulation = duplicate_theta_and_phi_values_for_simulation(theta_vector, phi_vector)


# ------------- FRUIT TRAJECTORY ------------------
def xy_by_fruit_trajectory(trajectory, total_time, dt):
    """
    returns vector of x and vector of y in the simulation dt interval in cm
    :param trajectory: function that gets double t and returns a tuple (x, y) of the fruit location by the estimated
    trajectory
    :param total_time: the total time of the fruit on the screen (from the moment that the trajectory was calculated)
    :param dt: the dt of the simulation
    :return: tuple of 2 lists (x of the fruit, y of the fruit) in the simulation dt interval in cm
    """
    # calculation of the right dt interval in order to show it in the simulation
    dt_trajectory = total_time / (T / dt)
    times = range(int(T / dt))
    x_fruit, y_fruit = [0 for _ in times], [0 for _ in times]
    for i in times:
        x_fruit[i], y_fruit[i] = trajectory(i * dt_trajectory)
        x_fruit[i], y_fruit[i] = Algo.algo_to_mech((x_fruit[i], y_fruit[i]))
        # # TODO the next 2 lines are a bit "fishy". check why it is necessary to add values for the conversion
        # x_fruit[i] += SCREEN[0] / 2
        # y_fruit[i] += d
    return x_fruit, y_fruit


def get_fruit_xy_vectors(fruits):
    def zero_trajectory(_):
        return 0, 0

    fruit_trajectories = [fruit.trajectory for fruit in fruits]

    # get the trajectory of the first fruit - (x,y) by t
    if len(fruit_trajectories) > 0:
        first_trajectory = []
        first_trajectory_total_time = []
        for i in range(len(fruit_trajectories)):
            # first_trajectory_object.append(fruit_trajectories[i])
            first_trajectory.append(fruit_trajectories[i].calc_trajectory())
            first_trajectory_total_time.append(fruit_trajectories[i].calc_life_time())
            #     # do not have to get into the 2 else down
            # else:
            #     first_trajectory = zero_trajectory
            #     first_trajectory_total_time = 1
    else:
        first_trajectory = []
        first_trajectory_total_time = 1

    xy_of_fruits_list = []
    for j in range(len(first_trajectory)):
        xy_of_fruits_list.append(xy_by_fruit_trajectory(first_trajectory[j],first_trajectory_total_time[j], dt_motor ))

    return xy_of_fruits_list

def init_multi():
    global SCREEN, d
    SCREEN = (12.0,8.0)
    d = 17.8


# ------------- CALCULATE LOCATIONS -------------
def run_simulation(points_to_go_through, fruits_sliced):
    """
    Runs the simulation.
    :param func: function of (x,y)(t), route of slice
    :param fruits_trajectories_and_starting_times:
    """
    # print("points to go through:")
    # print(points_to_go_through)
    # draw_points(points_to_go_through)

    global theta_simulation, phi_simulation

    make_slice_by_trajectory(points_to_go_through, False)

    xy_of_fruits_list = get_fruit_xy_vectors(fruits_sliced)

    # ------------- PLOT -------------------
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    x_0, y_0 = SCREEN[0], 0

    times = len(theta_simulation)

    x_practical_vector = [0 for _ in range(times)]
    y_practical_vector = [0 for _ in range(times)]

    # loop of plot
    for i in range(times):
        # quiting option
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()

        # plotting the screen
        screen.fill(WHITE)
        plot_screen(screen)

        # draw fruits locations
        for k in range(len(xy_of_fruits_list)):
            x_fruit, y_fruit = xy_board_to_xy_simulation(xy_of_fruits_list[k][0][i], xy_of_fruits_list[k][1][i])
            draw_circle([x_fruit, y_fruit], 10, screen, GREEN)

        # real locations
        x_practical, y_practical = xy_by_theta_phi(theta_simulation[i], phi_simulation[i])
        x_practical, y_practical = xy_board_to_xy_simulation(x_practical, y_practical)
        x_practical_vector[i], y_practical_vector[i] = x_practical, y_practical
        x_link_practical, y_link_practical = xy_by_theta(theta_simulation[i], x_0)
        draw_circle([x_practical, y_practical], 2, screen, RED)
        draw_line([x_link_practical, y_link_practical], [x_0, y_0], screen)
        draw_line([x_link_practical, y_link_practical], [x_practical, y_practical], screen)

        # display the simulation
        pygame.display.flip()

        # sleep for the simulation dt (dt_motor is the simulation dt)
        time.sleep(dt_motor)

    # quiting the simulation
    time.sleep(TIME_TO_QUIT_SIMULATION)
    pygame.display.quit()


if __name__ == '__main__':
    for i in range(1):
        run_simulation([(0.6,0.0), (-7.0,4.0), (7.0,4.0), (0.6,0.0)], [])
        time.sleep(1)
