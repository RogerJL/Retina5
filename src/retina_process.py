import dv_processing as dv
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

SYN_WEIGHT = 0.99
DISCHARGE = 0.70  # 1/tau_m
ACTIVATION_THRESHOLD = 1
RESET_LEVEL = -0.1
INHIBIT_WEIGHT = -2

MAX_DT = 500  # Timestamp ticks

class SpikingNeurons:
    def __init__(self, shape,
                 syn_weight=SYN_WEIGHT, discharge=DISCHARGE, reset_level=RESET_LEVEL,
                 inhibit_weight=INHIBIT_WEIGHT, inhibit_x_offset=0, inhibit_y_delta=0):
        # Constants
        self._syn_weight = syn_weight
        self._discharge = discharge
        self._reset_level = reset_level
        self._inhibit_weight = inhibit_weight

        # Open any camera
        self._neurons = np.zeros(shape=(shape[0] + abs(inhibit_x_offset), shape[1] + 2 * inhibit_y_delta))
        self._x_offset = -inhibit_x_offset if inhibit_x_offset < 0 else 0
        self._y_offset = inhibit_y_delta

        # Output to next
        self._spike_store = dv.EventStore()
        self._inhibit_x_offset = inhibit_x_offset
        self._inhibit_y_delta = inhibit_y_delta

        # TODO: handle with accumulator?
        #cv.namedWindow("Preview", cv.WINDOW_NORMAL)
        #cv.imshow("Preview", neurons.T)
        self._fig, self._ax = plt.subplots(1, 1)

        # Run the event processing while the camera is connected
        self._plot_timestamp = 0
        self._prev_timestamp = dv.now()

    def init_callback(self, first_timestamp):
        self._prev_timestamp = first_timestamp

    def dynamic_simulation_step(self, next_ts):
        dt = 1e-6 * (next_ts - self._prev_timestamp)
        self._neurons -= dt * self._discharge * self._neurons
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
            v = self._neurons[e_x + self._x_offset, e_y + self._y_offset] + self._syn_weight
            if v > ACTIVATION_THRESHOLD:
                self._spike_store.push_back(event.timestamp(), e_x, e_y, True)
                v = 0
            self._neurons[e_x + self._x_offset, e_y + self._y_offset] = v
            if self._inhibit_x_offset:
                step = 1 if self._inhibit_x_offset < 0 else -1
                for dy in range(-self._inhibit_y_delta, self._inhibit_y_delta + 1):
                    for dx in range(self._inhibit_x_offset, -step, step):
                        self._neurons[e_x + self._x_offset + dx, e_y + self._y_offset + dy] += self._inhibit_weight
            next_timestamp = event.timestamp()
            # Discharge all
            if next_timestamp - self._prev_timestamp > MAX_DT:
                self.dynamic_simulation_step(next_timestamp)

        #        if len(spiking[0]) or len(spiking[1]):
        #            print(prev_timestamp, spiking)

        # Discharge
        self.dynamic_simulation_step(next_timestamp)

        # Display
        if False:  # next_timestamp - self._plot_timestamp > 1e6:
            # Generate and show a preview of recent tracking history
            # cv.imshow("Preview", self._neurons.T)
            plt.imshow(self._neurons.T)
            plt.title("Right" if self._inhibit_x_offset < 0 else "Left" if self._inhibit_x_offset > 0 else "Input")
            plt.pause(0.01)

            self._plot_timestamp = next_timestamp

    def generateEvents(self):
        events = self._spike_store
        self._spike_store = dv.EventStore()
        return events
