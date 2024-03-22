from typing import List

import dv_processing as dv
import cv2 as cv

SKIP_N=2

class SlicingVisualizer:
    def __init__(self, shape):
        self._visualizer = dv.visualization.EventVisualizer(shape)
        self._skip = 0

    def show_comparision(self, events_lists: List[dv.EventStore]):
        self._skip -= 1
        if self._skip > 0:
            return
        self._skip = SKIP_N
        images = map(self._visualizer.generateImage, events_lists)
        cv.imshow("events", cv.vconcat(list(images)))
        cv.waitKey(1)
