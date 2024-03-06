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

layer1 = SpikingNeurons(shape=shape, syn_weight=0.1, tau_syn=0.1, tau_m=0.1)
layer_left = SpikingNeurons(shape=shape, syn_weight=3, inhibit_weight=-1.6, tau_syn=0.2, tau_m=0.5, inhibit_x_offset=3, inhibit_y_delta=1)
layer_right = SpikingNeurons(shape=shape, syn_weight=3, inhibit_weight=-1.6, tau_syn=0.2, tau_m=0.5, inhibit_x_offset=-3, inhibit_y_delta=1)

#labels = open("labels.txt", "wt")

# Register a callback every xx milliseconds
visualizer = SlicingVisualizer(shape)
def slicer_callback(events: dv.EventStore):
#    visualizer.show_comparision(events, events)

    layer_right.accept(events)
    right_events = layer_right.generateEvents()

    layer_left.accept(events)
    left_events = layer_left.generateEvents()

    left_ = len(left_events)
    right_ = len(right_events)
    if right_ or left_:
        first = (left_events.getLowestTime() if right_ == 0 else
                 right_events.getLowestTime() if left_ == 0 else
                 max(left_events.getLowestTime(), right_events.getLowestTime()))
        last = max(left_events.getHighestTime(), right_events.getHighestTime())
        out = f"{first:20d}\t{last:20d}\t{last-first:7d} \tleft_events: {left_:3d} \tright_events: {right_ :3d} \t{'Left' if left_ > max(10, 2 * right_) else 'Right' if right_ > max(10, 2 * left_) else '-'}"
        print(out)
        #labels.write(out + "\n")
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
#        slicer_callback(events)

#labels.close()