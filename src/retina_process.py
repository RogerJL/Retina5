import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

EVENT_SPIKE = 1
DISCHARGE = 0.99
THRESHOLD = 5
MAX_DT = 1000  # Timestamp ticks

class SpikingNeurons:
    def __init__(self):
        # Open any camera
        self._neurons = np.zeros(shape=(346, 260))

        # Initialize neurons with something visible
        self._neurons[:,25] = EVENT_SPIKE
        self._neurons[25,:] = EVENT_SPIKE

        self._spike_store = dv.EventStore()

        self._slicer = dv.EventStreamSlicer()


        #cv.namedWindow("Preview", cv.WINDOW_NORMAL)
        #cv.imshow("Preview", neurons.T)
        self._fig, self._ax = plt.subplots(1, 1)

        # Run the event processing while the camera is connected
        self._plot_timestamp = 0
        self._prev_timestamp = dv.now()

    def init_callback(self, first_timestamp):
        self._prev_timestamp = first_timestamp

    @classmethod
    def discharge(cls, prev_ts, next_ts, neuron_layer):
        dt = 1e-6 * (next_ts - prev_ts)
        neuron_layer -= dt * DISCHARGE * neuron_layer
        return next_ts

    # Declare the callback method for slicer
    def slicing_callback(self, events: dv.EventStore):
        # Pass events into the accumulator and generate a preview frame
        for event in events:
            # Inputs
            e_x = event.x()
            e_y = event.y()
            e_p = event.polarity()
            # Ignore bad pixels
            if e_x == 231 and e_y == 202:
                continue
            self._neurons[e_x, e_y] += EVENT_SPIKE
            # Self over threshold?
            #            if neurons[e_x, e_y] > THRESHOLD:
            #                print(event.timestamp(), e_x, e_y)
            next_timestamp = event.timestamp()
            if next_timestamp - self._prev_timestamp > MAX_DT:
                spiking = np.where(self._neurons > THRESHOLD)
                self._neurons[spiking] = 0
                for x, y in zip(spiking[0], spiking[1]):
                    self._spike_store.push_back(next_timestamp, x, y, True)
                self._slicer.accept(self._spike_store)
                self._prev_timestamp = SpikingNeurons.discharge(self._prev_timestamp, next_timestamp, self._neurons)

        spiking = np.where(self._neurons > THRESHOLD)
        self._neurons[spiking] = 0
        for x, y in zip(spiking[0], spiking[1]):
            self._spike_store.push_back(next_timestamp, x, y, True)
        self._slicer.accept(self._spike_store)
        #        if len(spiking[0]) or len(spiking[1]):
        #            print(prev_timestamp, spiking)

        # Discharge
        self._prev_timestamp = SpikingNeurons.discharge(self._prev_timestamp, next_timestamp, self._neurons)

        # Display
        if next_timestamp - self._plot_timestamp > 1e6:
            # Generate and show a preview of recent tracking history
            # cv.imshow("Preview", self._neurons.T)
            plt.imshow(self._neurons.T)
            plt.show()

            self._plot_timestamp = next_timestamp
            print(f"\r{self._plot_timestamp}   ", end='')
