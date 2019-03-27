"""
Takes care of communication with Arduino through Serial, including quantization of slice.
"""

import serial
from serial import SerialException
import math
import time
import numpy as np
# import matplotlib.pyplot as plt


# all lengths are in cm, all angles are in degrees
# (0,0) is in the middle of bottom side of screen


# ------------- CONSTANTS --------------

# MECHANICS CONSTANTS
DIMS = (16.0, 12.0)  # (X,Y)
STEPS_PER_REVOLUTION = 200  # number of full steps to make a full round
STEPS_FRACTION = 8  # the number of steps to make a full step (full step is 1.8 degrees)
MINIMAL_ANGLE = 2 * np.pi / (STEPS_PER_REVOLUTION * STEPS_FRACTION)  # the minimal angle step of the motor in rad (it is
# 1.8 degrees divided by the steps fraction)
ARMS = 15.0, 10.0  # length of arm links in cm
d = 14.7  # distance from motor to screen in cm
START_SLICE_LENGTH = 4

# SERIAL CONSTANTS
END_WRITING = '}'
START_SLICE = '~'
LENGTH_OF_COMMAND = 2  # the length of a command to the serial that contains the number of steps for each motor and the
SERIAL_BUFFER_SIZE = 64  # in bytes
COMMAND_PACKAGE_SIZE = math.floor(SERIAL_BUFFER_SIZE/LENGTH_OF_COMMAND)  # number of commands to write at once
STEPS_IN_COMMAND = 6  # number of steps to move at every command
MAX_STEPS_IN_COMMAND = 60  # max number of steps to move at every command


# TIME CONSTANTS
T = 1.0  # max value of parameter at slice
SERIAL_BPS = 115200  # the bit rate of reading and writing to the Serial
BITS_PER_BYTE = 8  # the number of bits in one byte
# direction of moving for each motor
WRITE_DELAY = 1000/(SERIAL_BPS/BITS_PER_BYTE/LENGTH_OF_COMMAND)  # delay in ms after writing to prevent buffer overload
TRAJECTORY_DIVISION_NUMBER = 20  # the number of parts that the trajectory of the arm is divided to
DT_DIVIDE_TRAJECTORY = float(T) / TRAJECTORY_DIVISION_NUMBER  # size of step in parameter
WANTED_RPS = 1.4  # speed of motors in revolutions per second
WANTED_RPS_SLOW = 0.1  # speed of motors in revolutions per second
ONE_STEP_DELAY = 5.0 / WANTED_RPS / STEPS_FRACTION * 2  # in ms
ONE_STEP_DELAY_SLOW = 5.0 / WANTED_RPS_SLOW / STEPS_FRACTION * 2  # in ms
ONE_STEP_DELAY_AVERAGE = (ONE_STEP_DELAY + ONE_STEP_DELAY_SLOW) / 2  # in ms
WAIT_FOR_STOP = 50.0  # time to wait after slice until committing invert slice in ms
STEPS_FOR_ACCELERATION = int(STEPS_FRACTION * 2 * WANTED_RPS)  # number of steps to move at acceleration move
if STEPS_FOR_ACCELERATION > MAX_STEPS_IN_COMMAND: STEPS_FOR_ACCELERATION = MAX_STEPS_IN_COMMAND
NUMBER_OF_ACCELERATION_MOVES = 1


# not in use
# RADIUS = 15
# ALPHA_MIN = (180/math.pi)*math.acos(DIMS[0]/(2.0*RADIUS))
# ALPHA_MAX = 180 - ALPHA_MIN
# STEPS_IN_CUT = STEPS_PER_REVOLUTION / 360.0 * (ALPHA_MAX - ALPHA_MIN)
# dt = 0.003      # the basic period of time of the simulation in sec
# times = int(T / dt)  # the size of the vectors for the simulation

try:
    ser = serial.Serial('com4', SERIAL_BPS)  # Create Serial port object
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


def make_slice_by_trajectory(points, invert=True):
    """
    Sends commands to Arduino according to the given route from the algorithmic module.
    :param points: list of tuples, each tuple is a point the arm should go through
    :param invert: if true then make also invert slice
    """
    steps_phi, steps_theta = generate_steps_from_points(points)
    move_2_motors(steps_theta, steps_phi)
    if invert:
        i_steps_theta, i_steps_phi = invert_slice(steps_theta, steps_phi)
        move_2_motors(i_steps_theta, i_steps_phi, True)


def generate_steps_from_points(points):
    steps_theta, steps_phi = list(), list()
    for i in range(len(points) - 1):
        current_point = xy2angles(points[i])  # in (theta,phi)
        next_point = xy2angles(points[i + 1])  # in (theta,phi)
        current_steps_theta, current_steps_phi = generate_steps_list(rad2steps(next_point[0] - current_point[0]),
                                                                     rad2steps(next_point[1] - current_point[1]))
        for j in range(len(current_steps_theta)):
            steps_theta.append(current_steps_theta[j])
            steps_phi.append(current_steps_phi[j])
    return steps_phi, steps_theta


def steps_in_slice_same_loop(steps_theta, steps_phi):
    """
    Calculates the number of iterations the slice will take when moving both motors in same loop.
    """
    output = 0
    for i in range(len(steps_theta)):
        output += max(steps_theta[i], steps_phi[i])
    return output


def steps_in_slice_different_loops(steps_theta, steps_phi):
    """
    Calculates the total number of steps in the slice.
    """
    return abs_sum(steps_theta) + abs_sum(steps_phi)


def abs_sum(lst):
    """
    Calculates the sum of the absolute values of the list elements
    """
    return sum([abs(num) for num in lst])


def xy2angles(point):
    """
    Converts coordinates from (x,y) to (theta,phi)
    :param point: tuple of (x,y)
    :return: tuple of (theta, phi)
    """
    x, y = point
    r = math.sqrt(x**2 + (d + y)**2)  # distance from main axis
    alpha = math.atan((d + y)/x) if x != 0 else math.pi/2  # angle between r and x axis
    if x < 0:
        alpha += math.pi
    # a = -1 + (1 + (-ARMS[0]**2 - ARMS[1]**2 + r**2) * (1.0 / (2 * ARMS[0] * ARMS[1]))) % 2
    # beta = math.acos(a)
    beta = math.acos((ARMS[0]**2 + ARMS[1]**2 - r**2) / (2 * ARMS[0] * ARMS[1]))  # angle between arms
    # b = -1 + (1 + (ARMS[0]**2 - ARMS[1]**2 + r**2) * (1.0 / (2 * ARMS[0] * r))) % 2
    # delta = math.acos(b)  # angle between r and 1st arm
    delta = math.acos((r**2 + ARMS[0]**2 - ARMS[1]**2) / (2 * r * ARMS[0]))
    theta = alpha + delta  # angle between 1st arm and x axis
    # phi = theta - beta  # angle between 2nd arm and x axis
    phi = beta - (math.pi - theta)

    return theta, phi


def angles2xy(point):
    """
    Converts coordinates from (theta,phi) to (x,y)
    :param point: tuple of (theta,phi)
    :return: tuple of (x,y)
    """
    theta, phi = point
    return ARMS[0] * math.cos(theta) + ARMS[1] * math.cos(phi), ARMS[0] * math.sin(theta) + ARMS[1] * math.sin(phi) - d


def wait(t_ms):
    """
    Creates a delay in the code. DO NOT USE WHILE THREADING, OR IT WILL GET STUCK!!!
    :param t_ms: time to wait in ms.
    """
    start = time.perf_counter()
    while time.perf_counter() < start + t_ms/1000.0:
        pass


def encode_message(steps_theta, steps_phi):
    """
    Builds the message to send to Arduino according to protocol.
    :param steps_theta: steps to move in theta motor
    :param steps_phi: steps to move in phi motor
    :return: '[steps-theta-bit][steps-phi-bit]'
    """
    return chr(steps_theta + 64) + chr(steps_phi + 64)


def move_2_motors(steps_theta, steps_phi, inverse=False):  # WRITE MAXIMUM 41 STEPS PER SLICE TODO check if inverse
    # needed
    """
    Sends commands to Arduino given the lists of steps.
    :param steps_theta: list of steps in theta
    :param steps_phi: list of steps in phi
    :param inverse: True if this is an inverse slice, False otherwise
    """

    # t1 = time.perf_counter()
    # print("Divide trajectory to " + str(len(steps_theta)) + " parts")

    # send trajectory to Arduino
    total_message = ""  # save the whole message for debugging
    for i in range(math.floor(len(steps_theta)/COMMAND_PACKAGE_SIZE)):  # send messages in packages
        message = ""
        for j in range(COMMAND_PACKAGE_SIZE):
            index = i * COMMAND_PACKAGE_SIZE + j
            message += encode_message(steps_theta[index], steps_phi[index])
        ser.write(str.encode(message))
        time.sleep(0.001 * COMMAND_PACKAGE_SIZE * WRITE_DELAY)
        total_message += message
    # send last package
    message = ""
    for i in range(len(steps_theta) - len(steps_theta) % COMMAND_PACKAGE_SIZE, len(steps_theta)):
        message += encode_message(steps_theta[i], steps_phi[i])
    ser.write(str.encode(message))
    # time.sleep(0.001*WRITE_DELAY*(len(steps_theta) % COMMAND_PACKAGE_SIZE))
    time.sleep(0.001 * len(message) * WRITE_DELAY)
    total_message += message
    # t2 = time.perf_counter()
    # print("time for writing: ", t2-t1)
    ser.write(str.encode(END_WRITING))
    total_message += END_WRITING

    # if it is an inverse slice, wait to prevent drifting
    # if inverse and time.perf_counter() < t1 + WAIT_FOR_STOP:
    #     time.sleep(0.001 * (WAIT_FOR_STOP + t1 - time.perf_counter()))

    # commit slice
    # print("CUT THEM!!!")
    ser.write(str.encode(START_SLICE))
    total_message += START_SLICE
    print("Theta steps:")
    print(str(steps_theta) + str(sum(steps_theta)))
    print("Phi steps:")
    print(str(steps_phi) + str(sum(steps_phi)))
    # print("message to write in serial: ")
    # print(total_message)
    # time_of_slice = ((abs_sum(steps_theta) + abs_sum(steps_phi)) * ONE_STEP_DELAY)
    time_of_slice = calc_time_of_slice(steps_theta, steps_phi)
    time.sleep(0.001 * time_of_slice)

    if len(total_message) > SERIAL_BUFFER_SIZE:
        print("BIG HUGE GIGANTIC EPIC WARNING - beware buffer overflow. length of message: " + str(len(total_message)))
    # wait for slice to end
    # time_of_slice = calc_time_of_slice(steps_theta, steps_phi)
    # time.sleep(0.001 * time_of_slice)
    # additional sleep
    # ser.write(str.encode(total_message))
    # time.sleep(1)

    # print steps made


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


def add_padding_for_acceleration(steps_list, padding_steps):
    """
    Add padding at beginning and end of steps list, to allow acceleration.
    :param steps_list: list of steps without padding
    :param padding_steps: total amount of steps for padding
    :return: padded list of steps
    """
    padding_list = break_into_equal_steps(padding_steps, STEPS_FOR_ACCELERATION)
    for i in range(len(padding_list)):
        if i % 2 == 0:
            steps_list = [padding_list[i]] + steps_list
        else:
            steps_list.append(padding_list[i])
    return steps_list


def phi_steps_by_theta_steps_for_acceleration(delta_steps_phi, steps_theta):
    steps_phi = []
    for move in steps_theta:
        if delta_steps_phi > move:
            steps_phi.append(move)
            delta_steps_phi -= move
        else:
            break
    if delta_steps_phi < STEPS_IN_COMMAND:
        if delta_steps_phi > 0:
            steps_phi.append(delta_steps_phi)
    else:
        end_of_steps_phi = break_into_equal_steps(delta_steps_phi, MAX_STEPS_IN_COMMAND)
        steps_phi = steps_phi + end_of_steps_phi
    return steps_phi


def break_into_equal_steps2(delta_steps_phi, num_of_commands):
    steps = [0 for _ in range(num_of_commands)]
    while delta_steps_phi > 0:
        steps[delta_steps_phi % num_of_commands] += 1
        delta_steps_phi -= 1
    return steps


def generate_steps_list_same_loop(delta_steps_theta, delta_steps_phi):
    """
    Generates lists of steps to move in each angle, given the total delta.
    :param delta_steps_theta: total delta of steps to move in theta
    :param delta_steps_phi: total delta of steps to move in phi
    :return: (steps_theta, steps_phi), tuple of lists of steps in each motor
    """
    theta_sign = sign(delta_steps_theta)
    phi_sign = sign(delta_steps_phi)
    delta_steps_theta = abs(delta_steps_theta)
    delta_steps_phi = abs(delta_steps_phi)
    delta_steps_theta_without_acceleration = delta_steps_theta - 2 * NUMBER_OF_ACCELERATION_MOVES * \
                                             STEPS_FOR_ACCELERATION
    if delta_steps_theta_without_acceleration < 0:
        delta_steps_theta_without_acceleration = 0
    padding_steps_theta = delta_steps_theta - delta_steps_theta_without_acceleration

    steps_theta_without_acceleration = break_into_equal_steps(delta_steps_theta_without_acceleration, STEPS_IN_COMMAND)

    steps_theta = add_padding_for_acceleration(steps_theta_without_acceleration, padding_steps_theta)

    steps_phi = phi_steps_by_theta_steps_for_acceleration(delta_steps_phi, steps_theta)

    steps_theta = add_zeros_at_end(steps_theta, len(steps_phi))
    steps_phi = add_zeros_at_end(steps_phi, len(steps_theta))

    if theta_sign < 0: steps_theta = [-steps for steps in steps_theta]
    if phi_sign < 0: steps_phi = [-steps for steps in steps_phi]

    # print("steps_theta: ")
    # print(steps_theta)
    # print("steps_phi: ")
    # print(steps_phi)

    return steps_theta, steps_phi


def generate_steps_list(delta_steps_theta, delta_steps_phi):
    """
    Generates lists of steps to move in each angle, given the total delta.
    :param delta_steps_theta: total delta of steps to move in theta
    :param delta_steps_phi: total delta of steps to move in phi
    :return: (steps_theta, steps_phi), tuple of lists of steps in each motor
    """
    theta_sign = sign(delta_steps_theta)
    phi_sign = sign(delta_steps_phi)
    delta_steps_theta = abs(delta_steps_theta)
    delta_steps_phi = abs(delta_steps_phi)

    steps_phi = break_into_equal_steps(delta_steps_phi, MAX_STEPS_IN_COMMAND)

    delta_steps_theta_without_acceleration = delta_steps_theta - 2 * NUMBER_OF_ACCELERATION_MOVES * \
                                             STEPS_FOR_ACCELERATION
    if delta_steps_theta_without_acceleration < 0:
        delta_steps_theta_without_acceleration = 0
    padding_steps_theta = delta_steps_theta - delta_steps_theta_without_acceleration

    steps_theta_without_acceleration = break_into_equal_steps(delta_steps_theta_without_acceleration, STEPS_IN_COMMAND)
    if len(steps_theta_without_acceleration) < len(steps_phi)-2*NUMBER_OF_ACCELERATION_MOVES:
        steps_theta_without_acceleration = break_into_equal_steps2(delta_steps_theta, len(steps_phi)-2*NUMBER_OF_ACCELERATION_MOVES)
    steps_theta = add_padding_for_acceleration(steps_theta_without_acceleration, padding_steps_theta)

    if len(steps_theta) > len(steps_phi): steps_phi = break_into_equal_steps2(delta_steps_phi, len(steps_theta))

    if theta_sign < 0: steps_theta = [-steps for steps in steps_theta]
    if phi_sign < 0: steps_phi = [-steps for steps in steps_phi]

    return steps_theta, steps_phi


def calc_time_of_slice(steps_theta, steps_phi):
    """
     Calculates the duration of the given slice. IN MILISECONDS
     :param steps_theta: steps of slice in theta
     :param steps_phi: steps of slice in phi
     :return: duration of given slice in ms
     """
    steps_in_slice = steps_in_slice_different_loops(steps_theta, steps_phi)

    time_of_slice = (steps_in_slice - STEPS_FOR_ACCELERATION * 2 * NUMBER_OF_ACCELERATION_MOVES) * ONE_STEP_DELAY + \
                    STEPS_FOR_ACCELERATION * 2 * NUMBER_OF_ACCELERATION_MOVES * ONE_STEP_DELAY_AVERAGE
    return time_of_slice

#     steps_counter = 20  # take spare
#     for i in range(len(steps_theta)):
#         steps_counter += abs(steps_theta[i]) + abs(steps_phi[i])
#     time_of_slice = steps_counter * ONE_STEP_DELAY
#     # print("time of slice is supposed to be " + str(time_of_slice/1000) + " seconds")
#     return time_of_slice


def break_into_steps(total_steps, step_per_command):
    """
    Splits the total amount of steps into portions.
    :param total_steps: total amount of steps to move
    :param step_per_command: amount of steps in each command
    :return: list of steps
    """
    steps_array = []
    while step_per_command < total_steps:
        steps_array += [step_per_command]
        total_steps -= step_per_command
    if total_steps > 0:
        steps_array += [total_steps]
    return steps_array


def break_into_equal_steps(total_steps, command_steps):
    """
    Splits the total amount of steps into equal portions.
    :param total_steps: total amount of steps to move
    :param command_steps: max amount of steps in command of the equal commands
    :return: list of steps
    """
    if total_steps == 0:
        return []
    num_of_steps = math.ceil(total_steps / command_steps)
    return break_into_equal_steps2(total_steps, num_of_steps)


def add_zeros_at_end(array, length):
    """
    Adds zeros to the end of the given array to make it in the given length.
    :return: same array, with 0s at its end, in the given length
    """
    if len(array) < length:
        array += [0] * (length - len(array))
    return array


def add_ones_at_end(array, length):
    """
    Adds ones to the end of the given array to make it in the given length.
    :return: same array, with 1s at its end, in the given length
    """
    last = None
    if len(array) != 0:
        last = array.pop()
    if len(array) < length:
        array += [1] * (length - len(array))
    if last is not None:
        array += [last]
    return array


def rad2steps(angle):
    """
    Converts the given angle in radians to steps (returns integer).
    """
    return math.floor((1/MINIMAL_ANGLE) * angle)


def start_cut(arm_loc):
    """
    Moves the pen to slice the starting apple.
    :param arm_loc: arm location at beginning of cut in (x,y)
    :return: arm location at end of cut in (x,y)
    """
    tot_r = ARMS[0] + ARMS[1]
    angle_to_move = math.acos(1 - (START_SLICE_LENGTH/tot_r)**2)
    steps_to_move = rad2steps(angle_to_move)
    steps_theta, steps_phi = generate_steps_list(steps_to_move, steps_to_move)
    move_2_motors(steps_theta, steps_phi)  # slice apple
    steps_theta, steps_phi = generate_steps_list(0, -steps_to_move)
    move_2_motors(steps_theta, steps_phi)  # go back with phi angle, to allow slice calculation
    theta_0, _ = xy2angles(arm_loc)
    return angles2xy((theta_0 + angle_to_move, theta_0))


def init_multi_arduino_communication():
    global DIMS
    global d
    DIMS = (12.0, 8.0)
    d = 17.8


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
    for _ in range(10):
        # make_slice_by_trajectory([(0.6,0.0), (0.6, 2.0), (0.6,4.0), (0.6,7.0), (0.6,9.0), (0.6,7.0), (0.6,4.0), (0.6, 2.0), (0.6,0.0)], False)
        # make_slice_by_trajectory([(0.6,0.0), (0.6,9.0), (0.6,0.0)], False)
        make_slice_by_trajectory([(7.0,4.0), (-7.0,4.0), (7.0,4.0)], False)
        # time.sleep(1)
    # make_slice_by_trajectory([(5.0,0.6), (0.6,0.0)], False)
    # generate_steps_list(7, -70)
