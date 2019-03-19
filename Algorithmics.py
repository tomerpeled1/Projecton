"""
Previously named SliceCreator. This file is the brain of the algorithmic module - it gets data from image processing
and generates slices (x and y locations) for arduino.
the coordinates here is (generally speaking) (x,y) when the 0,0 is at bottom left of the flipped screen
(the parabola of fruits routes is smiling).
"""

import math
import matplotlib.pyplot as plt
import statistics as st
import SliceTypes
import time
import ArduinoCommunication
import Simulation as Slm
from threading import Thread
import threading

# ----------------- CONSTANTS -------------------
#first acc is measured, second is from fazkanoot
RELATIVE_ACC = 1.478  # from experiences we did it tracker program
# RELATIVE_ACC = 1.1    # from experiences we did it tracker program
CAMERA_FPS = 30  # frames per second
TIME_BETWEEN_2_FRAMES = 1.0 / CAMERA_FPS  # in sec
CROP_SIZE = (160, 480)  # (y,x) in pixels
FRAME_SIZE = (480, 640)  # (y,x) in pixels
SCREEN_SIZE = (12, 16)  # (y,x) in cm
ACC = RELATIVE_ACC * SCREEN_SIZE[0]
INTEGRATE_WITH_MECHANICS = False  # make True to send slices to ArduinoCommunication

on_screen_fruits = []
SIMULATE = True  # make True to activate simulation
slice_queue_lock = threading.Condition()
simulation_thread = None
slice_queue = []
arm_loc = 0, 0


# ------------- CONVERTING FUNCTIONS -------------
def pixel2cm(pix_loc):
    """
    :param pix_loc: a pixel in order (x,y).
    :return: (x coord of screen, y coord of screen) when we look at the screen from the opposite side.
            we look at the opposite side because the arm is looking at the screen from it's top
            and we look at it from the bottom
    """
    (j_coord_crop, i_coord_crop, t) = pix_loc
    i_coord_frame = FRAME_SIZE[0] + (- CROP_SIZE[0] + i_coord_crop)
    j_coord_frame = FRAME_SIZE[1] / 2 - CROP_SIZE[1] / 2 + j_coord_crop
    i_coord_screen = (float(i_coord_frame / FRAME_SIZE[0])) * SCREEN_SIZE[0]
    j_coord_screen = (1 - float(j_coord_frame / FRAME_SIZE[1])) * SCREEN_SIZE[1]
    return j_coord_screen, i_coord_screen, t  # (x,y,t)


def cm2pixel(cm_loc):
    """
    :param cm_loc: cm location in order (x,y)
    :return: pixel location in order (y, x, t)
    """
    x_coord_screen, y_coord_screen, t = cm_loc
    x_coord_frame = int(x_coord_screen * float(FRAME_SIZE[1]) / SCREEN_SIZE[1])
    y_coord_frame = int((1.0 - float(y_coord_screen / SCREEN_SIZE[0])) * FRAME_SIZE[0])
    return y_coord_frame, x_coord_frame, t


# --------------- TRAJECTORY CLASS ---------------
class Trajectory:
    """
    class of fruit trajectory
    """

    def __init__(self, x0, y0, v, theta):
        """
        initiates the parameters for the trajectory
        :param x0: by the formula
        :param y0: by the formula
        :param v: by the formula
        :param theta: by the formula
        """
        self.x0 = x0
        self.y0 = y0
        self.v = v
        self.theta = theta

    def calc_trajectory(self):
        """
        return the trajectory of the fruit by the its parameters
        :return: a function of (x, y) by t (0 is the time of (x0, y0) and calc_life_time() is the time of the end of
        the trajectory (not exactly the end))
        """

        def get_xy_by_t(t):
            """
            :param t: double time
            :return: tuple (x, y) in cm
            """
            return self.x_trajectory(t), self.y_trajectory(t)

        return get_xy_by_t

    def calc_peak(self):
        """
        calculates the time and the y of the peak
        :return: tuple (t, y) in sec, cm
        """
        t = self.v * math.sin(self.theta) / ACC
        return t, self.calc_trajectory()(t)

    def calc_life_time(self):  # TODO improve function to return the time until the fruit exits the screen
        """
        returns the time that the fruit gets to the symmetric location to the initial location (x0, y0)
        :return: double time in sec
        """
        t = 2 * self.v * math.sin(self.theta) / ACC
        return t

    def x_trajectory(self, t):
        """
        returns the x value according to the formula of free fall
        :param t: time
        :return: x value according to the formula of free fall in cm
        """
        return self.x0 + self.v * math.cos(self.theta) * t

    def y_trajectory(self, t):
        """
        returns the y value according to the formula of free fall
        :param t: time
        :return: y value according to the formula of free fall in cm
        """
        # return SCREEN_SIZE[0] - v * math.sin(theta) * t + 0.5 * ACC * t ** 2
        return self.y0 - self.v * math.sin(self.theta) * t + 0.5 * ACC * t ** 2


# -------------------- slicing functions ------------------
def update_fruits(fruits):
    """
    Updates on_screen_fruits according to fruits list acquired from image processing.
    :param fruits: list of Fruit objects - fruits to add to on_scree_fruits
    """
    fruits_locs = [[pixel2cm(pix_loc[0]) for pix_loc in fruit.centers] for fruit in fruits]
    # centers = [[center for center in fruit.centers] for fruit in fruits]
    # centers2 = [[cm2pixel(loc) for loc in fruit_locs] for fruit_locs in fruits_locs]
    fruit_trajectories = [get_trajectory_by_fruit_locations(fruit_locs) for fruit_locs in fruits_locs]
    on_screen_fruits.extend([[fruit_trajectories[i], fruits[i].time_created] for i in range(len(fruits))])
    # on_screen_fruits.extend(fruits)
    fruits[:] = []

    # fruit_trajectories = [get_trajectory(fruit_locs) for fruit_locs in fruits_locs]
    # on_screen_fruits.extend([[fruit_trajectories[i], fruits[i].time_created] for i in range(len(fruits))])


def create_slice():
    """
    returns a slice according to the fruits that are on the screen
    :return: tuple of (slice_trajectory, timer, t_peak, fruit_trajectories_and_starting_times)
    """
    return calc_slice(on_screen_fruits)


def do_slice(slice_trajectory):
    """
    activate the simulation or the arduino by the given trajectory
    :param slice_trajectory: function of the location (x, y) of the pen in cm
    """
    parametrization, timer, time_of_slice, fruits_trajectories = slice_trajectory
    time_to_slice = time_until_slice(timer, time_of_slice)
    # run simulation
    if SIMULATE:
        Slm.run_simulation(parametrization, fruits_trajectories)
    # run arduino
    else:
        ArduinoCommunication.make_slice_by_trajectory(parametrization, time_to_slice)

def update_and_slice(fruits):
    """
    Updates the fruits on the screen and than creates a slice and adds it to slice queue.
    :param fruits: The fruits extracted by the image processing module.
    """
    global slice_queue
    global slice_queue_lock
    update_fruits(fruits)  # Updates the fruits on screen with the fruits extracted by the image processing module.
    if slice_queue_lock.acquire(False):  # If the access to slice queue is available, it means we are not in the middle
        # of a slice and we can create a new one.
        if len(on_screen_fruits) > 0:  # If there are fruits on the screen we want to create a slice.
            new_slice = create_slice()
            slice_queue.append(new_slice)  # Add the new slice to slice queue.
        slice_queue_lock.notify()
        slice_queue_lock.release()  # Release the lock on the slice queue.


def rms(array):
    """
    calculates rms
    :return: root mean square
    """
    sum_squares = 0
    length = len(array)
    for i in range(length):
        sum_squares += array[i] ** 2
    mean = sum_squares / length
    return math.sqrt(mean)


def get_r_coords_by_xy_coords(x_coords, y_coords):
    """
    Calculates the distance between points on trajectory
    :param x_coords: x coordinates along trajectory
    :param y_coords: y coordinates along trajectory
    :return: list of length len(x_coords)-1 of the distances between points
    """
    r_coords = [0 for _ in range(len(x_coords) - 1)]
    for i in range(len(y_coords) - 1):
        r_coords[i] = math.sqrt((x_coords[i + 1] - x_coords[i]) ** 2 + (y_coords[i + 1] - y_coords[i]) ** 2)
    return r_coords


def get_trajectory_by_fruit_locations(fruit_locs):
    """
    creating a trajectory according to the locations of the fruit by fitting speed v0 and initial angle theta
    :param fruit_locs: 2d list of [x, y, t] of the fruit (locations of 1 fruit)
    :return: trajectory object with the fitted values for speed (v0) and angle (theta)
    """

    x_coords = [fruit_loc[0] for fruit_loc in fruit_locs]
    y_coords = [fruit_loc[1] for fruit_loc in fruit_locs]
    # t_coords = [fruit_loc[2] for fruit_loc in fruit_locs]  # times from image processing are not accurate for sure
    r_coords = get_r_coords_by_xy_coords(x_coords, y_coords)

    # plot the given fruit locations (not the trajectory)
    # plt.plot(x_coords, y_coords, 'bo')
    # plt.show()

    # values between last location to first location
    x_total = x_coords[-1] - x_coords[0] if x_coords[-1] - x_coords[0] != 0 else 0.00001
    # y_total = y_coords[-1] - y_coords[0]
    # t_total = (len(x_coords) - 1) * TIME_BETWEEN_2_FRAMES
    # r_total = math.sqrt(x_total ** 2 + y_total ** 2)

    # *****options for x0 and y0 values*****
    x0_mean = st.mean(x_coords) if x_total != 0 else 0.001  # to prevent division by zero
    y0_mean = st.mean(y_coords)
    # x0_last = x_coords[-1]
    # y0_last = y_coords[-1]
    # x0 = x_coords[0]
    # y0 = y_coords[0]

    # *****options for v0 value*****
    # r_total_real = abs((SCREEN_SIZE[0] / 3 / math.sin(theta)))  # 3 is because the screen is croped to third
    v_array = [0 for _ in range(len(r_coords))]
    vy_array = [0 for _ in range(len(r_coords))]
    vx_array = [0 for _ in range(len(r_coords))]
    for i in range(len(r_coords)):
        # v_array[i] = r_coords[i] / TIME_BETWEEN_2_FRAMES
        vx_array[i] = (x_coords[i + 1] - x_coords[i]) / TIME_BETWEEN_2_FRAMES

        cur_vy = (y_coords[i + 1] - y_coords[i]) / TIME_BETWEEN_2_FRAMES
        # now try to get the v_entry from current v
        vy = cur_vy - ACC * i * TIME_BETWEEN_2_FRAMES
        vy_array[i] = vy

    # v0_median = st.median(v_array)
    # v0_mean = st.mean(v_array)
    # v0_rms = rms(v_array)
    # vy_median = st.median(vy_array)
    vy_mean = st.mean(vy_array)
    vx_mean = st.mean(vx_array)
    v0_mean = math.sqrt(vy_mean ** 2 + vx_mean ** 2)

    # v0_by_vy_end_to_end = y_total / t_total / math.sin(theta)
    # v0_by_vy_mean = vy_mean / math.sin(theta)
    # v0_by_vy_median = vy_median / math.sin(theta)
    # v0_stupid = r_total_real / (10 * TIME_BETWEEN_2_FRAMES)
    # v0_start_to_end = r_total / t_total

    v0 = v0_mean

    # *****options for theta value*****
    theta_array = [0 for _ in range(len(x_coords) - 1)]
    for i in range(len(x_coords) - 1):
        if (x_coords[i + 1] - x_coords[i]) != 0:  # to prevent division by zero
            delta_x = (x_coords[i + 1] - x_coords[i])
        else:
            delta_x = 0.001
        # theta_array[i] = math.pi - math.atan((y_coords[i + 1] - y_coords[i]) / delta_x)
        if vx_mean == 0: vx_mean = 0.00000001
        theta_array[i] = math.pi - math.atan(vy_mean/vx_mean)

    # theta_median = st.median(theta_array)  # best theta
    theta_mean = st.mean(theta_array)
    # theta_start_to_end = math.pi - math.atan(y_total / x_total)
    theta = theta_mean

    # ***** more options for x0 and y0 values*****
    x0_array, y0_array = [],[]
    for i in range(len(x_coords) - 1):
        x0_array.append(x_coords[i] - TIME_BETWEEN_2_FRAMES*i * vx_mean)
        y0_array.append(y_coords[i] - TIME_BETWEEN_2_FRAMES * i * vy_mean)

    x0 = st.mean(x0_array)
    y0 = st.mean(y0_array)

    # the trajectory by the fitted values (the returned object)
    trajectory = Trajectory(x0, y0, v0, theta)

    # plot the trajectory
    draw_trajectory_matplotlib(trajectory, x_coords, y_coords)

    return trajectory


def draw_trajectory_matplotlib(trajectory, x_coords, y_coords):
    """
    plot the trajectory and the fruit locations
    :param trajectory: trajectory object
    :param x_coords: x values of 1 fruit
    :param y_coords: y values of 1 fruit
    """
    t_tot = 3
    dt = 0.1
    x_lim = 20
    y_lim = 25

    times = range(-int(t_tot / dt), int(t_tot / dt))
    xy = [[0 for _ in times], [0 for _ in times]]
    route = trajectory.calc_trajectory()
    for i in times:
        xy[0][i], xy[1][i] = route(dt * i)

    # plt.plot(xy[0], xy[1], 'ro')
    # plt.plot(x_coords, y_coords, 'bo')
    # plt.ylim(0, y_lim)
    # plt.xlim(0, x_lim)
    # plt.show()


def calc_slice(fruit_trajectories_and_starting_times):
    """
    return the chosen slice - function of (x, y) by t
    :param fruit_trajectories_and_starting_times:
    :return:
    """
    return SliceTypes.slice_to_peak(get_pen_loc(), fruit_trajectories_and_starting_times)


def get_pen_loc():
    """
    :return: the location of the pen (x, y) in cm
    """
    # location (3cm, 3cm) from the bottom-left corner
    x_location = -SCREEN_SIZE[1] / 2 + 3
    y_location = 3
    return x_location, y_location
    # return -SCREEN_SIZE[1]/2+3, 3  # location (3cm, 3cm) from the bottom-left corner


def time_until_peak(time_created, time_of_slice):
    """
    Calculates the time #define stp 2
#define dir 3
#define MS1 4
#define MS2 5
#define MS3 6
#define EN  7

#define stp_2 8
#define dir_2 9
#define MS1_2 10
#define MS2_2 11
#define MS3_2 12
#define EN_2  13

#define MAX_RPS 3 // maximum revolutions per second the motor can do
#define WANTED_RPS 0.3 // the rps that was calculated for fastest slice
#define STEPS_FRACTION 8 // the number of fractions for full step (full step is 1.8 degrees)
#define ONE_STEP_DELAY 5000.0 / WANTED_RPS / STEPS_FRACTION // minimum delay in microseconds to put in moving loop
#define SERIAL_BPS 115200
#define LENGTH_OF_COMMAND 6
#define BITS_PER_BYTE 8
#define READ_DELAY 1000/(SERIAL_BPS/BITS_PER_BYTE/LENGTH_OF_COMMAND) // delay in microseconds to ensure safe reading from Serial (theoretiacaly should be 833.3333)
#define WRITE_DELAY READ_DELAY // delay in microseconds
#define MAX_NUM_OF_COMMANDS_IN_CUT 410
#define END_OF_COMMANDS 'e'
#define DO 'd'
#define MAX_MICRO_DELAY 15000
#define SLICE_END_SIGNAL 'z'
#define WAIT_FOR_STOP 100 // ms

int steps[MAX_NUM_OF_COMMANDS_IN_CUT][2] = {0};

void setup() {
  Serial.begin(SERIAL_BPS);
  Serial.setTimeout(5);

  pinMode(stp, OUTPUT);
  pinMode(dir, OUTPUT);
  pinMode(MS1, OUTPUT);
  pinMode(MS2, OUTPUT);
  pinMode(MS3, OUTPUT);
  pinMode(EN, OUTPUT);

  digitalWrite(stp, LOW);
  digitalWrite(dir, LOW);
  digitalWrite(MS1, HIGH);
  digitalWrite(MS2, HIGH);
  digitalWrite(MS3, LOW);
  digitalWrite(EN, LOW);

  pinMode(stp_2, OUTPUT);
  pinMode(dir_2, OUTPUT);
  pinMode(MS1_2, OUTPUT);
  pinMode(MS2_2, OUTPUT);
  pinMode(MS3_2, OUTPUT);
  pinMode(EN_2, OUTPUT);

  digitalWrite(stp_2, LOW);
  digitalWrite(dir_2, LOW);
  digitalWrite(MS1_2, HIGH);
  digitalWrite(MS2_2, HIGH);
  digitalWrite(MS3_2, LOW);
  digitalWrite(EN_2, LOW);

  Serial.print("ONE STEP DELAY is: ");
  Serial.println(ONE_STEP_DELAY);
}

void loop() {
  // make simple move
//  Serial.println("Write to serial");
//  write_to_serial_simple_move_of_motors();

  // steps of each motor are updated to the 'steps' array in the main scope (first index is for theta, second index is for phi)
//  digitalWrite(stp, HIGH);
  int moves_num = readStepsFromSerial();
//  digitalWrite(stp, LOW);

  // wait for signal to begin the movement of the motors - 'd'
  Serial.println("Ready for action...");
  while (readSerial() != DO){
//    delayMicroseconds(READ_DELAY);
  }
//  Serial.println(SLICE_END_SIGNAL);


  // move the motors according to the steps that were read from the Serial = CUT THE *****!!!
  Serial.println("CUT THEM ALL!!!!");
  for (int i = 0; i < moves_num; i++) {
    move(steps[i][0],steps[i][1]);
  }

}

int readStepsFromSerial() {
  // collect data - foramt is sending one order at a time in this order - (10's digit tetha, 1's digit tetha, 10's digit phi, 1's digit tetha, sgn_tetha, sgn phi)
  // sgn_angel is 1 if minus and 0 if plus
  int index=0;
  for (char in=readSerial();in!=END_OF_COMMANDS;in=readSerial()) {
    steps[index][0] = 10*(in-'0') + (readSerial()-'0'); // get abs() of theta
    in = readSerial();
    steps[index][1] = 10*(in-'0') + (readSerial()-'0'); // get abs() of phi
    if (readSerial()-'0') { // add sign of theta
//      Serial.println("adding minus to theta");
      steps[index][0] = -steps[index][0];
    }
    if (readSerial()-'0') { // add sign of phi
//      Serial.println("adding minus to phi");
      steps[index][1] = -steps[index][1];
    }
//    Serial.print("steps of small motor (theta): ");
    Serial.println(steps[index][0]);
//    Serial.print("steps of big motor (phi): ");
    Serial.println(steps[index][1]);
    index++;
  }
  return index;
}

int readSerial() {
  int in = Serial.read();
  while (in == -1 || in == '\n') {
    in = Serial.read();
  }
  waitMicroseconds(READ_DELAY);
  return in;
}

void write_to_serial_simple_move_of_motors() {
  //  String steps_frac = (String)STEPS_FRACTION;
  //  int message_length = 6;
  int steps_num = 1;
  //  String message_unclockwise = steps_frac + "0" + steps_frac + "0" + "0" + "0";
  //  String message_clockwise = steps_frac + "0" + steps_frac + "0" + "1" + "1";
  //  String message_unclockwise = "808000";
  //  String message_clockwise = "808011";

  // 10 steps (full steps for each motor) - unclockwise and than clockwise
  for (int i = 0; i < steps_num; i++) {
    Serial.write("808000");
    waitMicroseconds(WRITE_DELAY);
  }
  for (int i = 0; i < steps_num; i++) {
    Serial.write("808011");
    waitMicroseconds(WRITE_DELAY);
  }
  Serial.write(END_OF_COMMANDS);
}

void waitMicroseconds(double microseconds) {
    // choose between delay and delayMicroseconds
    if (microseconds > MAX_MICRO_DELAY) {
      delay(microseconds / 1000);
    } else {
      delayMicroseconds(microseconds);
    }
}


// small motor = steps1, big motor = steps2
// positive steps number = unclockwise, negative steps number = clockwise
void move(int steps1, int steps2) { // for now just move floor(steps)
//  Serial.println("moving motors:");
//  Serial.println(steps1);
//  Serial.println(steps2);
//  Serial.print('\n');
  if (steps1 < 0){
    digitalWrite(dir, LOW);
    steps1 = -steps1;
  } else digitalWrite(dir, HIGH);
  if (steps2 < 0){
    digitalWrite(dir_2, LOW);
    steps2 = -steps2;
  } else digitalWrite(dir_2, HIGH);
  double time1 = (double) millis();
  for (int i=0; i<steps1; i++) {
    digitalWrite(stp, HIGH);
    waitMicroseconds(ONE_STEP_DELAY / 2);
    digitalWrite(stp, LOW);
    waitMicroseconds(ONE_STEP_DELAY / 2);
  }
  double time2 = (double) millis();
  for (int i=0; i<steps2; i++) {
    digitalWrite(stp_2, HIGH);
    waitMicroseconds(ONE_STEP_DELAY / 2);
    digitalWrite(stp_2, LOW);
    waitMicroseconds(ONE_STEP_DELAY / 2);
  }
  double totTime = time2-time1; // in ms
//  Serial.print("It took ");
//  Serial.print(totTime);
//  Serial.println(" ms to make the move in theta");

//  double rps = double(steps)/(200*totTime);
//  Serial.print("We did ");
//  Serial.print(rps);
//  Serial.print(" rps. We intended to do ");
//  Serial.print(MAX_RPS);
//  Serial.println(" rps.");
}needs to be waited until slicing the fruit.
    :param fruit: fruit to calculate when to slice.
    :return: the time until the slice.
    """
    if time_created is None:
        print ("a")
    return time_created + time_of_slice - time.clock()

def time_until_slice(time_created, time_of_slice):
    return time_until_peak(time_created, time_of_slice)


def init_info(frame_size, crop_size=CROP_SIZE, screen_size=SCREEN_SIZE):
    """
    Initializes the sizes for the screen so that the algorithmics work properly.
    :param frame_size: size of frame in pixels
    :param crop_size: size of cropped frame in pixels
    :param screen_size: size of screen in cm
    """
    global CROP_SIZE, FRAME_SIZE, SCREEN_SIZE
    CROP_SIZE = crop_size
    FRAME_SIZE = frame_size
    SCREEN_SIZE = screen_size


def remove_sliced_fruits(fruits):
    """
    Remove fruits which have already been sliced.
    :param fruits: list of fruits to remove from on_screen_fruits (sliced fruits are not on the screen anymore).
    """
    for fruit in fruits:
        on_screen_fruits.remove(fruit)
    for fruit in on_screen_fruits:
        traj, timer = fruit
        if time.clock() > timer + traj.calc_life_time():
            on_screen_fruits.remove(fruit)


def mechanics_thread_run():
    """
    The function which runs in a different thread and executes the slices.
    """
    global slice_queue_lock
    global slice_queue
    while True:
        # Unlocks access to the slice queue (slices waiting to be done).
        slice_queue_lock.acquire()
        # Waits for a slice to enter the queue.
        while len(slice_queue) == 0:
            slice_queue_lock.wait()
        # Retrieves a slice for the queue.
        next_slice = slice_queue[0]
        slice_queue.remove(next_slice)
        # Executes slice (still memory not unlocked so that we want start a new slice during the previous one).
        do_slice(next_slice)
        # Release the access to the memory so that we can enter new slices to queue.
        slice_queue_lock.release()


def init_everything(integrate_with_mechanics=INTEGRATE_WITH_MECHANICS, simulate=SIMULATE):
    """
    Initializes the algorithmics module - opens a thread for the mechanics module.
    :param integrate_with_mechanics: boolean that decides weather to integrate with mechanics or not.
    :param simulate: boolean that decides weather to activate simulation or not.
    """
    global INTEGRATE_WITH_MECHANICS
    INTEGRATE_WITH_MECHANICS = integrate_with_mechanics
    global SIMULATE
    SIMULATE = simulate
    # In case we want to integrate with mechanics (simulation or arduino) we must open a new thread for it.
    if INTEGRATE_WITH_MECHANICS:
        global simulation_thread
        simulation_thread = Thread(target=mechanics_thread_run)
        simulation_thread.start()
    else:
        pass


if __name__ == "__main__":
    slice_and_times = create_slice()
    while True:
        do_slice(slice_and_times)
