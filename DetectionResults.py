import time

class DetectionResults:

    def __init__(self, contours, rectangles, centers):
        self.conts = contours
        self.rects = rectangles
        self.time_created = time.clock()
        self.centers = [(center[0], center[1], self.time_created) for center in centers]

    def pop_element(self, i):
        self.conts.pop(i)
        self.centers.pop(i)
        self.rects.pop(i)