import dv_processing as dv
from datetime import timedelta
from retina_process import SpikingNeurons
from retina_visualizer import SlicingVisualizer
import matplotlib.pyplot as plt

# Input store:
reader = dv.io.MonoCameraRecording("data/dvSave-2024_02_22_15_26_03.aedat4") # cameraSerial)

# Process store: Initialize a slicer
slicer = dv.EventStreamSlicer()

shape = (346, 200)
polarity = dv.EventPolarityFilter(False)
region = dv.EventRegionFilter((0, 0, shape[0], shape[1]))  # Bad pixels around (231, 202)
output_events = dv.EventStore()

layer1 = SpikingNeurons(shape=shape)
layer_left = SpikingNeurons(shape=shape, syn_weight=2, inhibit_x_offset=3, inhibit_y_delta=1)
layer_right = SpikingNeurons(shape=shape, syn_weight=2, inhibit_x_offset=-3, inhibit_y_delta=1)

# Register a callback every xx milliseconds
visualizer = SlicingVisualizer(shape)
def slicer_callback(event: dv.EventStore):
    layer_right.accept(events)
    right_events = layer_right.generateEvents()

    layer_left.accept(events)
    left_events = layer_left.generateEvents()

    print(f"left_events: {len(left_events):3d}, right_events: {len(right_events):3d}")
    visualizer.show_comparision(left_events, right_events)


slicer.doEveryTimeInterval(timedelta(milliseconds=20), slicer_callback)

# Find start time
while reader.isRunning():
    # Receive events
    events = reader.getNextEventBatch()

    # Check if anything was received
    if events is not None:
        # If so, pass the events into the slicer to handle them
        first_timestamp = events.getLowestTime()
        layer1.init_callback(first_timestamp)
        layer_left.init_callback(first_timestamp)
        layer_right.init_callback(first_timestamp)
        break

# Process
while reader.isRunning():
    # Receive events
    events = reader.getNextEventBatch()

    #image = reader.getNextFrame()
    #plt.imshow(image)
    #plt.pause(0.001)

    # Check if anything was received
    if events is not None:
        polarity.accept(events)
        events = polarity.generateEvents()

        region.accept(events)
        events = region.generateEvents()

        layer1.accept(events)
        events = layer1.generateEvents()

        # If so, pass the events into the slicer to handle them
        slicer.accept(events)
