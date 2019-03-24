import time
import Algorithmics as Algo


ARM_DELAY = 0
AVERAGE_SLICE_TIME = 0.5
CRITICAL_TIME = 0.5


class State:

    """
    State of the game - knows everything about all fruits on screen.
    """

    def __init__(self):
        self.fruits_out_of_range = []
        self.fruits_in_range = []
        self.current_time = time.perf_counter()
        self.arm_loc = Algo.get_pen_loc()

    def update_state(self, new_fruits, current_time):
        """
        Updates the state after getting information from image processing, doesn't take care of sliced fruits.
        :param new_fruits: new fruits discovered.
        :param current_time: time of current frame (normalized with main).
        :return:
        """
        self.current_time = current_time
        new_fruits = [fruit for fruit in new_fruits if fruit.trajectory]
        print("before add",self.fruits_out_of_range)
        print("before add", self.fruits_in_range)
        self.add_new_fruits(new_fruits)
        print("after add", self.fruits_out_of_range)
        print("after add", self.fruits_in_range)
        self.remove_old_fruits()
        print("after remove", self.fruits_out_of_range)
        print("after remove", self.fruits_in_range)

    def is_good_to_slice(self):
        """
        Determines whether or not should we slice right now.
        :return: tuple - (True, slice, sliced_fruits) if the state is good, (False, None, []) otherwise.
        """
        critical_time_locs = [(fruit,loc) for (fruit, loc) in
                                self.get_fruit_locations(CRITICAL_TIME, self.fruits_in_range)
                                    if not Algo.in_range_for_slice(loc)]
        if len(self.fruits_in_range) > 1:
            # print("FRUIT IN RANGE!")
            slice, sliced_fruits = Algo.create_slice(self, 0)
            if not slice is None:
                return True, slice, sliced_fruits
            else:
                return False, None, []
        else:
            return False, None, []

    def get_fruit_locations(self, time, fruits):
        """
        Calculates the fruit locations in (time) seconds.
        :param time: time from now in seconds.
        :return: [(x1,y1), (x2,y2), (x3,y3), ...] list of locations for all fruits.
        """
        locs = []
        for fruit in fruits:
            t = time + self.current_time - fruit.time_created
            print(str(t))
            locs.append((fruit, fruit.trajectory.calc_trajectory()(t)))
        return locs

        # return [(fruit, fruit.trajectory.calc_trajectory()(time + self.current_time - fruit.time_created)) for fruit in
        #         fruits]

    def add_new_fruits(self, fruits):
        """
        Updates the fruits - transfers fruits from out of range into fruits in range and adds fruits to fruits out
         of range.
        :param fruits: new fruits to add.
        """

        fruits_out_of_range_locs = self.get_fruit_locations(0, self.fruits_out_of_range)
        fruits_gone_into_range = []
        for fruit,loc in fruits_out_of_range_locs:
            if Algo.in_range_for_slice(loc):
                fruits_gone_into_range.append(fruit)
        self.fruits_in_range.extend(fruits_gone_into_range)
        self.fruits_out_of_range = [fruit for fruit in self.fruits_out_of_range if fruit not in fruits_gone_into_range]
        self.fruits_out_of_range.extend(fruits)

    def remove_old_fruits(self):
        """
        Transfers fruits which have gone out of range from in range to the garbage
        """
        # fruit_out_of_range_locs = self.get_fruit_locations(0, self.fruits_out_of_range)
        # self.fruits_out_of_range = [fruit for (fruit, loc) in fruit_out_of_range_locs
        #                             if Algo.on_screen(loc) and not Algo.in_range_for_slice(loc)]
        # fruits_gone_out_of_range = [fruit for (fruit, loc) in fruits_in_range_locs if not Algo.in_range_for_slice(loc)]
        # self.fruits_out_of_range.extend(fruits_gone_out_of_range)
        fruits_in_range_locs = self.get_fruit_locations(0, self.fruits_in_range)
        fruits_out_of_range_locs = self.get_fruit_locations(0, self.fruits_out_of_range)
        self.fruits_in_range = [fruit for (fruit, loc) in fruits_in_range_locs if Algo.in_range_for_slice(loc)]
        for index in range(len(fruits_out_of_range_locs)):
            flag = Algo.on_screen(fruits_out_of_range_locs[index][1])
            if not flag:
                print("*************************", fruits_out_of_range_locs[index][0], "centers: ", fruits_out_of_range_locs[index][0].centers)
                f = fruits_out_of_range_locs[index][0]
                print("self.time: ", self.current_time, "center: ", fruits_out_of_range_locs[index][1])
                for t in [k*0.05 for k in range(40)]:
                    print("t = ", t, "center from trajectory: ", f.trajectory.calc_trajectory()(t))
                self.fruits_out_of_range.remove(fruits_out_of_range_locs[index][0])
        # self.fruits_out_of_range = [fruit for (fruit, loc) in fruits_out_of_range_locs if Algo.on_screen(loc)]


    def remove_sliced_fruits(self, sliced_fruits):
        """
        Removes sliced fruits.
        :param sliced_fruits: fruits to remove.
        """
        self.fruits_in_range = [fruit for fruit in self.fruits_in_range if fruit not in sliced_fruits]
