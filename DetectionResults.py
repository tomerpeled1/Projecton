import time

class DetectionResults:

    def __init__(self, contours, rectangles, centers):
        self.conts = contours
        self.rects = rectangles
        self.centers = centers
        self.time_created = time.clock()

    def pop_element(self, i):
        self.conts.pop(i)
        self.centers.pop(i)
        self.rects.pop(i)