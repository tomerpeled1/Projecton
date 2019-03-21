"""
Takes care of communication with Arduino through Serial, including quantization of slice.
"""

import serial
from serial import SerialException
import math
import time
import numpy as np
import matplotlib.pyplot as plt


# all lengths are in cm, all angles are in degrees
# (0,0) is in the middle of bottom side of screen


# ------------- CONSTANTS --------------

# MECHANICS CONSTANTS
DIMS = (16.0, 12.0)  # (X,Y)
STEPS_PER_REVOLUTION = 200  # number of full steps to make a full round
STEPS_FRACTION = 8  # the number of steps to make a full step (full step is 1.8 degrees)
MINIMAL_ANGLE = 2 * np.pi / (STEPS_PER_REVOLUTION * STEPS_FRACTION)  # the minimal angle step of the motor in rad (it is
# 1.8 degrees divided by the steps fraction)
ARMS = [15.0, 10.0]  # length of arm links in cm
d = 15.0  # distance from motor to screen in cm
START_SLICE_LENGTH = 4

# SERIAL CONSTANTS
END_WRITING = '}'
START_SLICE = '~'
LENGTH_OF_COMMAND = 2  # the length of a command to the serial that contains the number of steps for each motor and the
SERIAL_BUFFER_SIZE = 64  # in bytes
COMMAND_PACKAGE_SIZE = math.floor(SERIAL_BUFFER_SIZE/LENGTH_OF_COMMAND)  # number of commands to write at once
MAX_COMMAND_IN_INVERT = 10  # number of steps to move at every command in inverse slice

# TIME CONSTANTS
T = 1.0  # max value of parameter at slice
SERIAL_BPS = 115200  # the bit rate of reading and writing to the Serial
BITS_PER_BYTE = 8  # the number of bits in one byte
# direction of moving for each motor
WRITE_DELAY = 1000/(SERIAL_BPS/BITS_PER_BYTE/LENGTH_OF_COMMAND)  # delay in ms after writing to prevent buffer overload
TRAJECTORY_DIVISION_NUMBER = 20  # the number of parts that the trajectory of the arm is divided to
DT_DIVIDE_TRAJECTORY = float(T) / TRAJECTORY_DIVISION_NUMBER  # size of step in parameter
WANTED_RPS = 0.6  # speed of motors in revolutions per second
ONE_STEP_DELAY = 5.0 / WANTED_RPS / STEPS_FRACTION  # in ms
WAIT_FOR_STOP = 50.0  # time to wait after slice until committing invert slice in ms

# not in use
# RADIUS = 15
# ALPHA_MIN = (180/math.pi)*math.acos(DIMS[0]/(2.0*RADIUS))
# ALPHA_MAX = 180 - ALPHA_MIN
# STEPS_IN_CUT = STEPS_PER_REVOLUTION / 360.0 * (ALPHA_MAX - ALPHA_MIN)
# dt = 0.003      # the basic period of time of the simulation in sec
# times = int(T / dt)  # the size of the vectors for the simulation

try:
    ser = serial.Serial('com6', SERIAL_BPS)  # Create Serial port object
    time.sleep(2)  # wait for 2 seconds for the communication to get established
except SerialException:
    print("Didn't create serial.")


# ---------- ALGORITHMIC FUNCTION ---------------
def get_xy(t):  # gets time in sec
    """
    Example of an output from the algorithmic module (constant acceleration, constant speed, constant deceleration)
    :param t: parameter of route. 0<=t<=T
    :return: (x,y) in the given t.
    """
    acc = 1800.0
    x_0 = -DIMS[0] / 2
    y_0 = 0.8 * DIMS[1]
    d_a = DIMS[0] / 4.0
    t_a = math.sqrt(2 * d_a / acc)
    v = acc * t_a

    x = x_0
    y = y_0

    if t < t_a:
        x = x_0 + 0.5 * acc * math.pow(t, 2)
    elif t_a < t < T - t_a:
        x = x_0 + d_a + v * (t - t_a)
    elif t > T - t_a:
        x = x_0 + d_a + v * (T - 2 * t_a) - 0.5 * acc * math.pow(t - (T - t_a), 2)
    return x, y


# ----------- SLICE FUNCTIONS ----------------
def modulo(a, n):
    """
    Fixed modulo function
    :param a: first argument
    :param n: second argument
    :return: fixed a % n
    """
    if a >= 0:
        return a % n
    else:
        return a % n - 1


def quantize_trajectory(get_xy_by_t):
    """
    Samples the continuous get_xy_by_t to calculate steps_theta and steps_phi
    :param get_xy_by_t: continuous function given from algorithmic module
    :return: steps_theta, steps_phi, lists of steps to make in each motor
    """
    theta, phi = get_angles_by_xy_and_dt(get_xy_by_t, DT_DIVIDE_TRAJECTORY)
    d_theta, d_phi = np.diff(theta), np.diff(phi)
    steps_theta_decimal, steps_phi_decimal = ((1 / MINIMAL_ANGLE) * d_theta), ((1 / MINIMAL_ANGLE) * d_phi)
    for i in range(len(theta) - 2):
        steps_theta_decimal[i + 1] += modulo(steps_theta_decimal[i], 1)
        steps_phi_decimal[i + 1] += modulo(steps_phi_decimal[i], 1)
    steps_theta = steps_theta_decimal.astype(int)
    steps_phi = steps_phi_decimal.astype(int)
    for i in range(len(steps_phi)):
        if steps_phi[i] > 50 or steps_theta[i] > 50:
            a=0
    return steps_theta, steps_phi


def make_slice_by_trajectory(get_xy_by_t, time_to_slice):
    """
    Sends commands to Arduino according to the given route from the algorithmic module.
    :param get_xy_by_t: function given form algorithmic module
    :param time_to_slice: time to wait until the slice should be executed
    """
    steps_theta, steps_phi = quantize_trajectory(get_xy_by_t)
    wait(time_to_slice - calc_time_of_slice(steps_theta, steps_phi))
    move_2_motors(steps_theta, steps_phi)
    i_steps_theta, i_steps_phi = invert_slice(steps_theta, steps_phi)
    move_2_motors(i_steps_theta, i_steps_phi, True)


def get_angles_by_xy_and_dt(get_xy_by_t, dt):
    """
    Converts continuous function of (x,y)(t) to discrete lists of angles.
    :param get_xy_by_t: function given form algorithmic module
    :param dt: discretization of time
    :return: (theta, phi), tuple of lists
    """
    # sample function
    times = range(int(T / dt) + 1)
    # get xy by dt
    xy = [[0 for _ in times], [0 for _ in times]]
    for i in times:
        xy[0][i], xy[1][i] = get_xy_by_t(dt * i)

    # plot function
    # plt.plot(xy[0], xy[1])
    # plt.show()

    # calc angles by xy
    r = np.sqrt(np.power(xy[0], 2) + np.power(np.add(d, xy[1]), 2))  # distance from main axis
    alpha = np.arctan2(np.add(d, xy[1]), xy[0])  # angle between r and x axis
    a = np.add(-1, np.remainder(np.add(1, np.multiply(np.add(-math.pow(ARMS[0], 2) - math.pow(ARMS[1], 2),
                                                             np.power(r, 2)), 1.0 / (2 * ARMS[0] * ARMS[1]))), 2))
    beta = np.arccos(a)  # angle between arms
    b = np.add(-1, np.remainder(np.add(1, np.multiply(np.add(math.pow(ARMS[0], 2) - math.pow(ARMS[1], 2),
                                                             np.power(r, 2)), 1.0 / (2 * ARMS[0] * r))), 2))
    delta = np.arccos(b)  # angle between r and 1st arm
    theta = alpha + delta  # angle between 1st arm and x axis
    phi = theta - beta  # angle between 2nd arm and x axis

    for i in range(len(theta)-1):
        if abs(theta[i+1]-theta[i]) > 0.2 or abs(phi[i+1]-phi[i]) > 0.2:
            pass

    return theta, phi


def wait(t):
    """
    Creates a delay in the code. DO NOT USE WHILE THREADING, OR IT WILL GET STUCK!!!
    :param t: time to wait in ms.
    """
    start = time.perf_counter()
    while time.perf_counter() < start + t/1000.0:
        pass


def encode_message(steps_theta, steps_phi):
    """
    Builds the message to send to Arduino according to protocol.
    :param steps_theta: steps to move in theta motor
    :param steps_phi: steps to move in phi motor
    :return: '[steps-theta-bit][steps-phi-bit]'
    """
    return chr(steps_theta + 64) + chr(steps_phi + 64)


def move_2_motors(steps_theta, steps_phi, inverse=False):  # WRITE MAXIMUM 41 STEPS PER SLICE
    """
    Sends commands to Arduino given the lists of steps.
    :param steps_theta: list of steps in theta
    :param steps_phi: list of steps in phi
    :param inverse: True if this is an inverse slice, False otherwise
    """

    t1 = time.perf_counter()
    # print("Divide trajectory to " + str(len(steps_theta)) + " parts")

    # send trajectory to Arduino
    for i in range(math.floor(len(steps_theta)/COMMAND_PACKAGE_SIZE)):  # send messages in packages
        message = ""
        for j in range(COMMAND_PACKAGE_SIZE):
            index = i * COMMAND_PACKAGE_SIZE + j
            message += encode_message(steps_theta[index], steps_phi[index])
        ser.write(str.encode(message))
        time.sleep(0.001*COMMAND_PACKAGE_SIZE*WRITE_DELAY)
    # send last package
    message = ""
    for i in range(len(steps_theta) - len(steps_theta) % COMMAND_PACKAGE_SIZE, len(steps_theta)):
        message += encode_message(steps_theta[i], steps_phi[i])
    ser.write(str.encode(message))
    time.sleep(0.001*COMMAND_PACKAGE_SIZE*(len(steps_theta) % COMMAND_PACKAGE_SIZE))

    t2 = time.perf_counter()
    print("time for writing: ", t2-t1)
    ser.write(str.encode(END_WRITING))

    # if it is an inverse slice, wait to prevent drifting
    if inverse and time.perf_counter() < t1 + WAIT_FOR_STOP:
        time.sleep(0.001 * (WAIT_FOR_STOP + t1 - time.perf_counter()))

    # commit slice
    # print("CUT THEM!!!")
    ser.write(str.encode(START_SLICE))
    # wait for slice to end
    time_of_slice = calc_time_of_slice(steps_theta, steps_phi)
    # time.sleep(0.001 * time_of_slice)
    # additional sleep
    time.sleep(0.01)

    # print steps made
    print("Theta steps:")
    print(str(steps_theta) + str(sum(steps_theta)))
    print("Phi steps:")
    print(str(steps_phi) + str(sum(steps_phi)))


def sign(x):
    """
    Calculates sign of input.
    :param x: input number
    :return: 1 if x>0, -1 if x<0, 0 if x==0
    """
    return int(x/abs(x)) if x != 0 else 0


def invert_slice(steps_theta, steps_phi):
    """
    Calculates the slice for returning to start point.
    :param steps_theta: list of steps in theta in first slice
    :param steps_phi: list of steps in phi in first slice
    :return: (i_steps_theta, i_steps_phi), steps of returning-to-start slice
    """
    # print("INVERT SLICE")
    return generate_steps_list(-sum(steps_theta), -sum(steps_phi))


def generate_steps_list(delta_theta, delta_phi):
    """
    Generates lists of steps to move in each angle, given the total delta.
    :param delta_theta: total delta to move in theta
    :param delta_phi: total delta to move in phi
    :return: (steps_theta, steps_phi), tuple of lists of steps in each motor
    """
    steps_theta = [sign(delta_theta) * a for a in break_into_steps(abs(delta_theta), MAX_COMMAND_IN_INVERT)]
    steps_phi = [sign(delta_phi) * a for a in break_into_steps(abs(delta_phi), MAX_COMMAND_IN_INVERT)]
    steps_theta = add_zeros_at_end(steps_theta, len(steps_phi))
    steps_phi = add_zeros_at_end(steps_phi, len(steps_theta))
    return steps_theta, steps_phi


def calc_time_of_slice(steps_theta, steps_phi):
    """
    Calculates the duration of the given slice.
    :param steps_theta: steps of slice in theta
    :param steps_phi: steps of slice in phi
    :return: duration of given slice in ms
    """
    steps_counter = 20  # take spare
    for i in range(len(steps_theta)):
        steps_counter += abs(steps_theta[i]) + abs(steps_phi[i])
    time_of_slice = steps_counter * ONE_STEP_DELAY
    # print("time of slice is supposed to be " + str(time_of_slice/1000) + " seconds")
    return time_of_slice


def break_into_steps(total_steps, maximal_step):
    steps_array = []
    while maximal_step < total_steps:
        steps_array += [maximal_step]
        total_steps -= maximal_step
    if total_steps > 0:
        steps_array += [total_steps]
    return steps_array


def add_zeros_at_end(array, length):
    if len(array) < length:
        array += [0] * (length - len(array))
    return array


def start_cut():
    """
    Moves the pen to slice the starting apple.
    :return: arm_loc at end of cut in (x,y)
    """
    tot_r = ARMS[0] + ARMS[1]
    angle_to_move = math.acos(1 - (START_SLICE_LENGTH/tot_r)**2)
    steps_theta, steps_phi = generate_steps_list(angle_to_move, angle_to_move)
    move_2_motors(steps_theta, steps_phi)  # slice apple
    steps_theta, steps_phi = generate_steps_list(0, -angle_to_move)
    move_2_motors(steps_theta, steps_phi)  # go back with phi angle, to allow slice calculation
    theta_0 = math.acos(DIMS[0]/(2*tot_r))  # theta of arm at beginning
    return -ARMS[0]*math.cos(theta_0 + angle_to_move) - ARMS[1]*math.cos(theta_0),\
        ARMS[0]*math.sin(theta_0 + angle_to_move) + ARMS[1]*math.sin(theta_0)


# if __name__ == "__main__":
# print('Lets begin...')
# make_slice_by_trajectory(get_xy)
# move_2_motors([99], [0])
# print("starting")
# st1 = [5 for i in range(41)]
# st2 = [5 for i in range(41)]
# move_2_motors(st1, st2)
# start = time.time()
# for i in range(int(25000)):
#     pass
# print(time.time()-start)

if __name__ == '__main__':
    # ערס mode
    while True:
        steps_theta_main = 36 * [-5]
        steps_phi_main = 36 * [-10]
        move_2_motors(steps_theta_main, steps_phi_main)
        start_main = time.perf_counter()
        i_steps_theta_main, i_steps_phi_main = invert_slice(steps_theta_main, steps_phi_main)
        while 1000.0*(time.perf_counter() - start_main) < WAIT_FOR_STOP:
            pass
        move_2_motors(i_steps_theta_main, i_steps_phi_main)
        wait(calc_time_of_slice(steps_theta_main, steps_phi_main))
