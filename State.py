import time
import Algorithmics as Algo


ARM_DELAY = 0
AVERAGE_SLICE_TIME = 0.5


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
        self.add_new_fruits(new_fruits)
        self.remove_old_fruits()
        self.current_time = current_time

    def is_good_to_slice(self):
        """
        Determines whether or not should we slice right now.
        :return: tuple - (True, slice, sliced_fruits) if the state is good, (False, None, []) otherwise.
        """
        if self.fruits_in_range:
            slice_to_return = Algo.create_slice(self, 0)
            return True, slice_to_return, self.fruits_in_range
        else:
            return False, None, []

    def get_fruits_locations(self, time_from_now, fruits):
        """
        Calculates the fruit locations in (time) seconds.
        :param time_from_now: time from now in seconds.
        :param fruits: list of fruits to calculate locations
        :return: [(x1,y1), (x2,y2), (x3,y3), ...] list of locations for all fruits.
        """
        return [(fruit, fruit.trajectory.calc_trajectory()(time_from_now + self.current_time - fruit.time_created))
                for fruit in fruits]

    def add_new_fruits(self, fruits):
        """
        Updates the fruits - transfers fruits from out of range into fruits in range and adds fruits to fruits out
         of range.
        :param fruits: new fruits to add.
        """
        fruits_out_of_range_locs = self.get_fruits_locations(0, self.fruits_out_of_range)
        self.fruits_in_range = [fruit for (fruit, loc) in fruits_out_of_range_locs if Algo.in_range_for_slice(loc)]
        self.fruits_out_of_range.extend(fruits)

    def remove_old_fruits(self):
        """
        Transfers fruits which have gone out of range into fruits out of range and removes fruits which have left the
        screen.
        """
        fruit_out_of_range_locs = self.get_fruits_locations(0, self.fruits_out_of_range)
        self.fruits_out_of_range = [fruit for (fruit, loc) in fruit_out_of_range_locs
                                    if Algo.on_screen(loc) and not Algo.in_range_for_slice(loc)]
        fruits_in_range_locs = self.get_fruits_locations(0, self.fruits_in_range)
        fruits_gone_out_of_range = [fruit for (fruit, loc) in fruits_in_range_locs if not Algo.in_range_for_slice(loc)]
        self.fruits_out_of_range.extend(fruits_gone_out_of_range)
        self.fruits_in_range = [fruit for fruit in self.fruits_in_range if fruit not in fruits_gone_out_of_range]

    def remove_sliced_fruits(self, sliced_fruits):
        """
        Removes sliced fruits.
        :param sliced_fruits: fruits to remove.
        """
        self.fruits_in_range = [fruit for fruit in self.fruits_in_range if fruit not in sliced_fruits]
