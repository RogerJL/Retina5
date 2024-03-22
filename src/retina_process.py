from typing import Tuple

import dv_processing as dv
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

SYN_WEIGHT = 0.99
TAU = 1
ACTIVATION_THRESHOLD = 1
RESET_LEVEL = -0.1
INHIBIT_AREA = 2 * 3
INHIBIT_WEIGHT = -3 / INHIBIT_AREA

MAX_DT = 1000  # Timestamp ticks

def scale_image(array, limits: Tuple[int | float, int | float]) -> np.ndarray:
    array = np.maximum(array - limits[0], 0)
    array = np.minimum(1, array / (limits[1] - limits[0])) # 0..1
    return (255 * array).astype(np.uint8)

class SpikingNeurons:
    def __init__(self, shape,
                 syn_weight=SYN_WEIGHT, tau_syn=TAU, tau_m=TAU, reset_level=RESET_LEVEL,
                 inhibit_weight=INHIBIT_WEIGHT, inhibit_x_offset=0, inhibit_y_delta=0,
                 id=""):
        # Constants
        self._syn_weight = syn_weight
        self._discharge_syn = 1 / tau_syn
        self._discharge_m = 1 / tau_m
        self._reset_level = reset_level
        self._inhibit_weight = inhibit_weight
        self.R = 10

        # Open any camera
        self._neurons_i = np.zeros(shape=(shape[0] + abs(inhibit_x_offset), shape[1] + 2 * inhibit_y_delta),
                                   dtype=np.float32)
        self._x_offset = -inhibit_x_offset if inhibit_x_offset < 0 else 0
        self._y_offset = inhibit_y_delta
        self._neurons_v = np.zeros(shape=shape,
                                   dtype=np.float32)

        # Output to next
        self._spike_store = dv.EventStore()
        self._inhibit_x_offset = inhibit_x_offset
        self._inhibit_y_delta = inhibit_y_delta

        # Run the event processing while the camera is connected
        self.id = id
        self._plot_timestamp = 0
        self._prev_timestamp = dv.now()

    def init_callback(self, first_timestamp):
        self._prev_timestamp = first_timestamp

    def dynamic_simulation_step(self, next_ts):
        dt = next_ts - self._prev_timestamp
        if dt == 0:
            return
        dt *= 1e-6

        self._neurons_i -= min(1, dt * self._discharge_syn) * self._neurons_i
        vshape = self._neurons_v.shape
        self._neurons_v += (-min(1, dt * self._discharge_m) * self._neurons_v
                            + dt * self._discharge_m * self.R * self._neurons_i[self._x_offset:self._x_offset + vshape[0],
                                                                                self._y_offset:self._y_offset + vshape[1]])
        spiking = np.where(self._neurons_v > ACTIVATION_THRESHOLD)
        self._neurons_v[spiking] = RESET_LEVEL
        for x, y in zip(spiking[0], spiking[1]):
            self._spike_store.push_back(next_ts, x, y, True)

        self._prev_timestamp = next_ts

    # Declare the callback method for slicer
    def accept(self, events: dv.EventStore):
        next_timestamp = self._prev_timestamp
        # Pass events into the accumulator and generate a preview frame
        for event in events:
            # Inputs
            e_x = event.x()
            e_y = event.y()
            # unused e_p = event.polarity()
            self._neurons_i[e_x + self._x_offset, e_y + self._y_offset] += self._syn_weight
            if self._inhibit_x_offset:
                step = 1 if self._inhibit_x_offset < 0 else -1
                self._neurons_i[e_x + self._x_offset + 1, e_y + self._y_offset] += self._syn_weight
                for dy in range(-self._inhibit_y_delta, self._inhibit_y_delta + 1):
                    for dx in range(self._inhibit_x_offset, -step, step):
                        self._neurons_i[e_x + self._x_offset + dx, e_y + self._y_offset + dy] += self._inhibit_weight
            next_timestamp = event.timestamp()

        # Dynamic simulation and spike detection
        self.dynamic_simulation_step(next_timestamp)

        # Display
        if True:  #next_timestamp - self._plot_timestamp > 1e6:
            # Generate and show a preview of recent tracking history
            neurons_i = self._neurons_i
            if self._x_offset or self._inhibit_y_delta:
                neurons_i = neurons_i[self._x_offset: , self._inhibit_y_delta:-self._inhibit_y_delta]
            images = [scale_image(neurons_i.T, (-10, 10)),
                      scale_image(self._neurons_v.T, (-ACTIVATION_THRESHOLD, ACTIVATION_THRESHOLD))]
            cv.imshow("neuron " + str(self.id), cv.resize(cv.hconcat(list(images)), dsize=(0,0), fx=2, fy=2))
            cv.waitKey(100)

            self._plot_timestamp = next_timestamp

    def generateEvents(self):
        events = self._spike_store
        self._spike_store = dv.EventStore()
        return events
