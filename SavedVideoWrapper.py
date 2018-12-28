import cv2


class SavedVideoWrapper:
    '''
    this class is a wrapper for a saved video, so the code can run not on live stream
    '''

    def __init__(self,src):
        '''
        :param src: string, the name of video file
        '''
        self.stream = cv2.VideoCapture(src) # need to have isOpened




    def read(self):
        _,frame = self.stream.read()
        return frame


    def stop(self):
        pass