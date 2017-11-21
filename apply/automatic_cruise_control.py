from sweeppy import Sweep

# A script that handles slowing down the vehicle when close to another vehicle ahead, using the Sweep LIDAR sensor
# Created by brendon-ai, November 2017

# The file path to the LIDAR device
LIDAR_DEVICE_PATH = '/dev/temp'

# Create a device using the constant path
with Sweep(LIDAR_DEVICE_PATH) as sweep:
    # Start scanning with the LIDAR
    sweep.start_scanning()
