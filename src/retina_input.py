import dv_processing as dv
from datetime import timedelta
from src.retina_process import init_callback, slicing_callback

# Input store:
reader = dv.io.MonoCameraRecording("../data/dvSave-2024_02_22_15_26_03.aedat4") # cameraSerial)

# Output store: Initialize a slicer
slicer = dv.EventStreamSlicer()

# Register a callback every 33 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)

# Find start time
while reader.isRunning():
    # Receive events
    events = reader.getNextEventBatch()

    # Check if anything was received
    if events is not None:
        # If so, pass the events into the slicer to handle them
        first_timestamp = events[0].timestamp()
        break

init_callback(first_timestamp)

while reader.isRunning():
    # Receive events
    events = reader.getNextEventBatch()

    # Check if anything was received
    if events is not None:
        # If so, pass the events into the slicer to handle them
        slicer.accept(events)
