from picamera2 import Picamera2
import numpy as np
import time
from picamera2.encoders import Encoder
import matplotlib.pyplot as plt
import threading, queue
from zero_suppression_encoding import  encode_frame, make_file_header

MODE = "source" # "source", "contiguous_frames", "video"
TIME =1.5  # seconds of video in video mode
FRAME_RATE = 50  # frames per second in video mode
N_FRAMES = 100000 # number of frames to capture in frames mode
FILENAME = "output/source_sr90_gain12_compressed"
FORMAT = 'SGBRG10'  # Bayer format, 10 bits per pixel stored in 16 bits
EXPOSURE_TIME = 20000  # in microseconds
ANALOGUE_GAIN = 12.0  # Gain factor max 22.3
CROP = (0.3, 0.3, 1., 1. )# (0.5, 0.5, 0.25, 0.25)  # Relative crop (x, y, width, height) 
# The foolowing settings are not used for raw capture, but left here for completeness
COLOUR_GAINS = (2.5, 2.1)  # (Red gain, Blue gain) Should be 1,1 for raw?
CONTRAST = 1.0 #Range 0.0 (low contrast) to 32.0 (high contrast)
NOISE_REDUCTION_MODE = "Off" # Off, Fast, HighQuality
SATURATION = 1.0 # Range 0.0 (monochrome) to 32.0 (highly saturated)
SHARPNESS = 0.0 # Range 0.0 (no_sharpening) to 16.0 (sharp)
AWB = False  # Automatic white balance
AE = False  # Automatic exposure
# Parameters for zero suppression and encoding
ZS_THRESHOLD = 70  # Zero suppression threshold
QUEUE_SIZE = 500  # Size of the frame queue for contiguous frames mode

picam2 = Picamera2()
size = picam2.sensor_modes[0]['size']
full_resolution = picam2.sensor_modes[-1]['size']
print(f"Using sensor mode with size: {size}")
print (f"Controls: {picam2.sensor_modes}")
video_config = picam2.create_video_configuration(
               raw={"format": FORMAT, "size": size},
               controls={"ExposureTime": EXPOSURE_TIME, "AnalogueGain": ANALOGUE_GAIN, "FrameRate":FRAME_RATE, "AeEnable":AE, "AwbEnable":AWB} )

picam2.configure(video_config)


roi = (int(CROP[0]*full_resolution[0]), int(CROP[1]*full_resolution[1]), int(CROP[2]*size[0]), int(CROP[3]*size[1]))
print (f"Setting ROI to: {roi}")
picam2.set_controls({"ScalerCrop": roi})
print (f"Configured camera with video config: {video_config}")

picam2.start()
time.sleep(1)  # Let camera warm up

timestamps = []
frame_width = video_config['raw']['size'][0]
frame_height = video_config['raw']['size'][1]
stride = video_config['raw']['stride']
gain = video_config['controls']['AnalogueGain']
exposure = video_config['controls']['ExposureTime']
if MODE == "video":
    def save_timestamps(request):
        timestamps.append(request.get_metadata()['SensorTimestamp'])

    picam2.encode_stream_name = "raw"
    encoder = Encoder()

    picam2.pre_callback = save_timestamps
    print(f"Starting recording to {FILENAME}.raw for {TIME} seconds")
    picam2.start_recording(encoder, f"{FILENAME}.raw", pts='timestamp.txt')
    time.sleep(TIME)
    print("Stopping recording")
    start_time = time.time()
    picam2.stop_recording()
    end_time = time.time()
    print(f"Recorded video for {TIME} seconds, processing took {end_time - start_time:.2f} seconds")

elif MODE == "contiguous_frames":
    mm = np.memmap(f"{FILENAME}.dat", dtype=np.uint8, mode='w+', shape=(N_FRAMES, frame_height, stride))
    q = queue.Queue(maxsize=10)
    done = False

    def writer_thread(batch_size=100):
        idx = 0
        batch = []

        while not done or not q.empty():
            try:
                frame = q.get(timeout=1)
            except queue.Empty:
                continue

            if frame is None:
                break

            batch.append(frame)

            # When enough frames have been collected
            if len(batch) >= batch_size:
                # Stack into one array (batch_size, H, W, C)
                block = np.stack(batch, axis=0)

                # Write to memmap slice all at once
                mm[idx:idx + len(block)] = block
                idx += len(block)

                # Flush data to disk
                mm.flush()

                # Reset batch list
                batch.clear()

        # Write remaining frames in last partial batch
        if batch:
            block = np.stack(batch, axis=0)
            mm[idx:idx + len(block)] = block
            mm.flush()
    
    # Start writer thread
    start_time = time.time()
    thread = threading.Thread(target=writer_thread)
    thread.start()

    for i in range(N_FRAMES):
        print(f"taking {i}")
        request = picam2.capture_request()
        raw_image = request.make_array("raw")
        q.put(raw_image)
        request.release()
    end_time = time.time()
    done = True
    thread.join()
    picam2.stop()
    print(f"Captured {N_FRAMES} frames in {end_time - start_time:.2f} seconds, fps: {N_FRAMES / (end_time - start_time):.2f}")

elif MODE == "source":
    outfile = open(f"{FILENAME}.raw", "wb")
    file_header = make_file_header(frame_width, frame_height)
    outfile.write(file_header.tobytes())
    q_frames = queue.Queue(maxsize=QUEUE_SIZE)
    q_timestamps = queue.Queue(maxsize=QUEUE_SIZE)
    done = False
    def compression_and_writer_thread():
        while not done or not q_frames.empty():
            try:
                raw_image = q_frames.get(timeout=1)
                timestamp = q_timestamps.get(timeout=1)
            except queue.Empty:
                continue
            header, packed = encode_frame(raw_image, timestamp, ZS_THRESHOLD)
            outfile.write(header.tobytes())
            packed.tofile(outfile)

    start_time = time.time()
    thread = threading.Thread(target=compression_and_writer_thread)
    thread.start()
    alredy_written = False
    for i in range(N_FRAMES):
        print(f"taking {i}")
        request = picam2.capture_request()
        raw_frame = request.make_array("raw").view(np.uint16)[:, :frame_width]
        timestamp = request.get_metadata()['SensorTimestamp']
        q_frames.put(raw_frame)
        q_timestamps.put(timestamp)
        request.release()
        # Check queue size and wait if necessary
        if q_frames.qsize() >= QUEUE_SIZE:
            if not alredy_written:
                intermediate_time = time.time()
                print(f"Filled up queue at frame: {i}, after {intermediate_time - start_time:.2f} seconds, fps: {i / (intermediate_time - start_time):.2f}")
                alredy_written = True
    end_time = time.time()
    done = True
    thread.join()
    picam2.stop()
    print(f"Captured {N_FRAMES} frames in {end_time - start_time:.2f} seconds, fps: {N_FRAMES / (end_time - start_time):.2f}")