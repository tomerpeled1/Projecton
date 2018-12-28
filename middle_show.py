import serial
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import SliceCreator


# all lengths are in cm, all angles are in degrees
# (0,0) is in the middle of bottom side of screen
# Initiate arm at MAXIMAL theta


# ------------- SETUP THE SERIAL -------------
ser = serial.Serial('com3', 9600)  # Create Serial port object
time.sleep(2)  # wait for 2 seconds for the communication to get established


# inpt = input("Enter 1 for stupid slice")
# while inpt != "1":
#     inpt = input()
# SliceCreator.make_slice()


# ------------- CONSTANTS --------------
# board constants
RADIUS = 15
DIMS = (21.7, 13.6)  # (X,Y)
ALPHA_MIN = (180/math.pi)*math.acos(DIMS[0]/(2.0*RADIUS))
ALPHA_MAX = 180 - ALPHA_MIN
STEPS_PER_REVOLUTION = 200
STEPS_FRACTION = 8
MINIMAL_ANGLE = 2 * np.pi / (STEPS_PER_REVOLUTION * STEPS_FRACTION)
STEPS_IN_CUT = STEPS_PER_REVOLUTION / 360.0 * (ALPHA_MAX - ALPHA_MIN)
ARMS = [15, 10]     # length of arm links in cm
d = 18
# time constants
T = 1          # total time of slice - it is not real time but parametrization
WRITE_DELAY = 1  # delay in ms after writing to prevent buffer overload
# dt = 0.003      # the basic period of time of the simulation in sec
# times = int(T / dt)  # the size of the vectors for the simulation
TRAJECTORY_DIVISION_NUMBER = 40
DT_DIVIDE_TRAJECTORY = float(T) / TRAJECTORY_DIVISION_NUMBER
END_WRITING = 'e'
START_SLICE = 'd'


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
    Fixed modulo function (TODO add explanation)
    :param a: first argument
    :param n: second argument
    :return: fixed a % n
    """
    if a > 0:
        return a % n
    else:
        return a % n - 1


def make_slice_by_trajectory(get_xy_by_t):
    """
    Sends commands to Arduino according to the given route from the algorithmic module.
    :param get_xy_by_t: function given form algorithmic module
    """
    theta, phi = get_angles_by_xy_and_dt(get_xy_by_t, DT_DIVIDE_TRAJECTORY)
    d_theta, d_phi = np.diff(theta), np.diff(phi)
    steps_theta_decimal, steps_phi_decimal = ((1 / MINIMAL_ANGLE) * d_theta), ((1 / MINIMAL_ANGLE) * d_phi)
    for i in range(len(theta)-2):
        steps_theta_decimal[i+1] += modulo(steps_theta_decimal[i], 1)
        steps_phi_decimal[i+1] += modulo(steps_phi_decimal[i], 1)
    steps_theta = steps_theta_decimal.astype(int)
    steps_phi = steps_phi_decimal.astype(int)
    move_2_motors(steps_theta, steps_phi)


def get_angles_by_xy_and_dt(get_xy_by_t, dt):
    """
    Converts continuous function of (x,y)(t) to discrete lists of angles.
    :param get_xy_by_t: function given form algorithmic module
    :param dt: discretization of time
    :return: {theta, phi), tuple of lists
    """
    times = range(int(T / dt))
    # get xy by dt
    xy = [[0 for _ in times], [0 for _ in times]]
    for i in times:
        xy[0][i], xy[1][i] = get_xy_by_t(dt * i)

    plt.plot(xy[0], xy[1])
    # plt.show()

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

    return theta, phi


def wait(t):
    """
    Creates a delay in the code.
    :param t: time to wait in ms.
    """
    for i in range(int(25000*t)):
        pass


# theta - small motor.    phi - big motor
def move_2_motors(steps_theta, steps_phi):  # WRITE MAXIMUM 41 STEPS PER SLICE
    """
    Sends commands to Arduino given the lists of steps.
    :param steps_theta: list of steps in theta
    :param steps_phi: list of steps in phi
    """
    t1 = time.time()
    print("Divide trajectory to " + str(len(steps_theta)) + " parts")
    for i in range(len(steps_theta)):
        message = abs(steps_theta[i]) * 10000 + abs(steps_phi[i]) * 100
        if steps_theta[i] < 0:
            message += 10
        if steps_phi[i] < 0:
            message += 1

        message = str(message)
        len_message = len(message)
        for _ in range(6-len_message):
            message = "0" + message

        ser.write(str.encode(message))
        wait(WRITE_DELAY)
        # print(str(message))

    t2 = time.time()
    print("time for writing: ", t2-t1)
    ser.write(str.encode(END_WRITING))
    print("ended writing")
    # time.sleep(3)
    ser.write(str.encode(START_SLICE))

    print("Theta steps:")
    print(steps_theta)
    print("Phi steps:")
    print(steps_phi)


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

# if __name__ == '__main__':
#     print(1)
