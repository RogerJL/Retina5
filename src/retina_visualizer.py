import dv_processing as dv
import matplotlib.pyplot as plt
import cv2 as cv

SKIP_N=2

class SlicingVisualizer:
    def __init__(self, shape):
        self._visualizer = dv.visualization.EventVisualizer(shape)
        self._skip = 0
# perf issue        self._fig, self._axs = plt.subplots(2, 1)

    def show_comparision(self, left_events: dv.EventStore, right_events: dv.EventStore):
        self._skip -= 1
        if self._skip > 0:
            return
        self._skip = SKIP_N
        left_image = self._visualizer.generateImage(left_events)
        right_image = self._visualizer.generateImage(right_events)
# perf. issue       self._axs[0].imshow(left_image)
#        self._axs[0].title.set_text("Left")
#        self._axs[1].imshow(right_image)
#        self._axs[1].title.set_text("Right")
        plt.imshow(cv.vconcat([left_image, right_image]))  # slows down... can be closed - restarts fast
        plt.title("Left / Right")
        #plt.show(block=False)
        #plt.draw()
        plt.pause(0.001)