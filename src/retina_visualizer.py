import dv_processing as dv
import matplotlib.pyplot as plt
import cv2 as cv


class SlicingVisualizer:
    def __init__(self, shape):
        self._visualizer = dv.visualization.EventVisualizer(shape)

    def show_comparision(self, left_events: dv.EventStore, right_events: dv.EventStore):
        left_image = self._visualizer.generateImage(left_events)
        right_image = self._visualizer.generateImage(right_events)
        plt.imshow(cv.hconcat([left_image, right_image]))
        plt.title("Left <-----> Right")
        plt.show()
