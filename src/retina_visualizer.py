from typing import List

import dv_processing as dv
import matplotlib.pyplot as plt
import cv2 as cv

SKIP_N=2

class SlicingVisualizer:
    def __init__(self, shape):
        self._visualizer = dv.visualization.EventVisualizer(shape)
        self._skip = 0
# perf issue        self._fig, self._axs = plt.subplots(2, 1)

    def show_comparision(self, events_lists: List[dv.EventStore]):
        self._skip -= 1
        if self._skip > 0:
            return
        self._skip = SKIP_N
        images = map(self._visualizer.generateImage, events_lists)
# perf. issue       self._axs[0].imshow(left_image)
#        self._axs[0].title.set_text("Left")
#        self._axs[1].imshow(right_image)
#        self._axs[1].title.set_text("Right")
        plt.imshow(cv.vconcat(list(images)))  # slows down... can be closed - restarts fast
        plt.title("Input / Left / Right")
        #plt.show(block=False)
        #plt.draw()
        plt.pause(0.001)