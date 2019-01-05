"""
Defines the Fruit class, that represents a detected fruit.
"""

import random


class Fruit:

    def __init__(self, track_window, hist, counter, centers, time_created):
        self.track_window = track_window
        self.hist = hist
        self.counter = counter
        self.centers = centers
        self.time_created = time_created
        self.is_falling = False
        self.trajectory = None
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
