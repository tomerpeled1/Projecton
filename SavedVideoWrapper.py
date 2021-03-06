"""
Implements the SavedVideoWrapper class.
"""

import cv2
import time


class SavedVideoWrapper:
    """
    this class is a wrapper for a saved video, so the code can run not on live stream
    """

    def __init__(self, src):
        """
        :param src: string, the name of video file
        """
        self.stream = cv2.VideoCapture(src)  # need to have isOpened

    def read(self):
        """
        Gets a frame from the video.
        :return: frame from video
        """
        _, frame = self.stream.read()
        # cv2.waitKey(15)
        time.sleep(0.02)
        return frame

    def stop(self):
        pass
