import dv_processing as dv
from datetime import timedelta
from src.retina_process import SpikingNeurons

# Input store:
reader = dv.io.MonoCameraRecording("../data/dvSave-2024_02_22_15_26_03.aedat4") # cameraSerial)

# Output store: Initialize a slicer
slicer = dv.EventStreamSlicer()

layer1 = SpikingNeurons()

# Register a callback every 33 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=33), layer1.slicing_callback)

# Find start time
while reader.isRunning():
    # Receive events
    events = reader.getNextEventBatch()

    # Check if anything was received
    if events is not None:
        # If so, pass the events into the slicer to handle them
        first_timestamp = events[0].timestamp()
        layer1.init_callback(first_timestamp)
        break

# Process
while reader.isRunning():
    # Receive events
    events = reader.getNextEventBatch()

    # Check if anything was received
    if events is not None:
        # If so, pass the events into the slicer to handle them
        slicer.accept(events)
