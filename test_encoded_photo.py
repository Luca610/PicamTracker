#!/usr/bin/env python3

from picamera2 import Picamera2
import numpy as np
import time
from picamera.zero_suppression_encoding import  encode_frame, make_file_header, decode_hits, decode_header, file_header_dtype, frame_header_dtype
import matplotlib.pyplot as plt

FILENAME = "encoded_photos.raw"
FORMAT = 'SGBRG10'  # Bayer format, 10 bits per pixel stored in 16 bits
EXPOSURE_TIME = 1000000  # in microseconds
ANALOGUE_GAIN = 5.0  # Gain factor
COLOUR_GAINS = (2.5, 2.1)  # (Red gain, Blue gain) Should be 1,1 for raw?
CROP = (0.6, 0.3, 1., 1. )# (0.5, 0.5, 0.25, 0.25)  # Relative crop (x, y, width, height) 
FRAME_DURATION_LIMITS = (200, 200)  # in microseconds (min, max)

picam2 = Picamera2()
size = picam2.sensor_modes[0]['size']
full_resolution = picam2.sensor_modes[-1]['size']
print(f"Using sensor mode with size: {size}")
print (f"Controls: {picam2.sensor_modes}")
video_config = picam2.create_video_configuration(
               raw={"format": FORMAT, "size": size},
               controls={"ExposureTime": EXPOSURE_TIME, "AnalogueGain": ANALOGUE_GAIN} )

picam2.configure(video_config)

picam2.start()

print (f"Full configuration:", video_config)

time.sleep(0.5)  # allow sensor to stabilize
filename = "reference.jpg"
picam2.capture_file(filename)  
request = picam2.capture_request()
raw_image1 = request.make_array("raw").view(np.uint16)[:, :size[0]]
timestamp1 = request.get_metadata()['SensorTimestamp']
request.release()
time.sleep(0.5)  # allow sensor to stabilize
filename = "reference2.jpg"
picam2.capture_file(filename)
request = picam2.capture_request()
raw_image2 = request.make_array("raw").view(np.uint16)[:, :size[0]]
timestamp2 = request.get_metadata()['SensorTimestamp']
request.release()
picam2.stop()

with open(FILENAME, "wb") as outfile:
    file_header = make_file_header(size[0], size[1])
    outfile.write(file_header.tobytes())
    header, packed = encode_frame(raw_image1, timestamp1, 64)
    outfile.write(header.tobytes())
    packed.tofile(outfile)
    header, packed = encode_frame(raw_image2, timestamp2, 64)
    outfile.write(header.tobytes())
    packed.tofile(outfile)

print(f"Encoded frames written to {FILENAME}")

# For testing, plot the raw images before and after encoding + decododing
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Raw Image 1")
plt.imshow(raw_image1, cmap='gray', vmin=0, vmax=1024)
plt.subplot(1, 2, 2)
plt.title("Original Raw Image 2")
plt.imshow(raw_image2, cmap='gray', vmin=0, vmax=1024)
plt.savefig("original_raw_images.png")

# Decode the frames back from the file for verification
def decode_frame_from_file(filename, frame_index):
    with open(filename, "rb") as infile:
        file_header = np.frombuffer(infile.read(file_header_dtype.itemsize), dtype=file_header_dtype)[0]
        frame_width = file_header['frame_width']
        frame_height = file_header['frame_height']
        for i in range(frame_index + 1):
            frame_header = np.frombuffer(infile.read(frame_header_dtype.itemsize), dtype=frame_header_dtype)[0]
            timestamp, nhits = decode_header(frame_header)
            packed_data = np.frombuffer(infile.read(nhits * 4), dtype=np.uint32)
        x, y, amplitudes = decode_hits(packed_data)
        decoded_image = np.zeros((frame_height, frame_width), dtype=np.uint16)
        decoded_image[x, y] = amplitudes
        return decoded_image, timestamp
decoded_image1, decoded_timestamp1 = decode_frame_from_file(FILENAME, 0)
decoded_image2, decoded_timestamp2 = decode_frame_from_file(FILENAME, 1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Decoded Raw Image 1")
plt.imshow(decoded_image1, cmap='gray', vmin=0, vmax=1024)
plt.subplot(1, 2, 2)
plt.title("Decoded Raw Image 2")
plt.imshow(decoded_image2, cmap='gray', vmin=0, vmax=1024)
plt.savefig("decoded_raw_images.png")


    