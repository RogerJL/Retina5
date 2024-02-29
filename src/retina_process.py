import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

EVENT_SPIKE = 1
DISCHARGE = 0.99
THRESHOLD = 5
MAX_DT = 1000  # Timestamp ticks

# Open any camera
neurons = np.zeros(shape=(346, 260))

# Initialize neurons with something visible
neurons[:,25] = EVENT_SPIKE
neurons[25,:] = EVENT_SPIKE

spike_store = dv.EventStore()

slicer = dv.EventStreamSlicer()


#cv.namedWindow("Preview", cv.WINDOW_NORMAL)
#cv.imshow("Preview", neurons.T)
fig, ax = plt.subplots(1, 1)

def discharge(prev_ts, next_ts, neuron_layer):
    dt = 1e-6 * (next_ts - prev_ts)
    neuron_layer -= dt * DISCHARGE * neuron_layer
    return next_ts

# Declare the callback method for slicer
def slicing_callback(events: dv.EventStore):
    global next_timestamp, prev_timestamp, plot_timestamp, spike_store
    # Pass events into the accumulator and generate a preview frame
    for event in events:
        # Inputs
        e_x = event.x()
        e_y = event.y()
        e_p = event.polarity()
        # Ignore bad pixels
        if e_x == 231 and e_y == 202:
            continue
        neurons[e_x, e_y] += EVENT_SPIKE
        # Self over threshold?
        #            if neurons[e_x, e_y] > THRESHOLD:
        #                print(event.timestamp(), e_x, e_y)
        next_timestamp = event.timestamp()
        if next_timestamp - prev_timestamp > MAX_DT:
            spiking = np.where(neurons > THRESHOLD)
            neurons[spiking] = 0
            for x, y in zip(spiking[0], spiking[1]):
                spike_store.push_back(next_timestamp, x, y, True)
            slicer.accept(spike_store)
            prev_timestamp = discharge(prev_timestamp, next_timestamp, neurons)

    spiking = np.where(neurons > THRESHOLD)
    neurons[spiking] = 0
    for x, y in zip(spiking[0], spiking[1]):
        spike_store.push_back(next_timestamp, x, y, True)
    slicer.accept(spike_store)
    #        if len(spiking[0]) or len(spiking[1]):
    #            print(prev_timestamp, spiking)

    # Discharge
    prev_timestamp = discharge(prev_timestamp, next_timestamp, neurons)

    # Display
    if next_timestamp - plot_timestamp > 1e6:
        # Generate and show a preview of recent tracking history
        # cv.imshow("Preview", neurons.T)
        plt.imshow(neurons.T)
        plt.show()

        plot_timestamp = next_timestamp
        print(f"\r{plot_timestamp}   ", end='')

# Run the event processing while the camera is connected
plot_timestamp = 0
prev_timestamp = next_timestamp = dv.now()

def init_callback(first_timestamp):
    global prev_timestamp, next_timestamp

    prev_timestamp = next_timestamp = first_timestamp
