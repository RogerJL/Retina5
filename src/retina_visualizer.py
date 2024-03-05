import dv_processing as dv
import matplotlib.pyplot as plt
import cv2 as cv

SKIP_N=2

class SlicingVisualizer:
    def __init__(self, shape):
        self._visualizer = dv.visualization.EventVisualizer(shape)
        self._skip = 0

    def show_comparision(self, left_events: dv.EventStore, right_events: dv.EventStore):
        self._skip -= 1
        if self._skip > 0:
            return
        self._skip = SKIP_N
        left_image = self._visualizer.generateImage(left_events)
        right_image = self._visualizer.generateImage(right_events)
        plt.imshow(cv.vconcat([left_image, right_image]))
        plt.title("Left / Right")
        #plt.show(block=False)
        #plt.draw()
        plt.pause(0.001)