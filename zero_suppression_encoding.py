

import numpy as np

MAGIC_WORD = 0xABCD1234

file_header_dtype = np.dtype([
        ('magic_word', np.uint32),
        ('frame_width', np.uint32),
        ('frame_height', np.uint32)
    ])
frame_header_dtype = np.dtype([
        ('magic_word', np.uint32),
        ('timestamp', np.float64),
        ('nhits', np.uint32)
    ])

def zero_suppression(array, threshold):
    """Apply zero suppression to the input array."""
    x, y = np.nonzero(array > threshold)
    amplitudes = array[x,y]
    return x, y, amplitudes

def encode_hits(x, y, amplitudes):
    """Encode hits into a compact 32-bit representation."""
    packed_data = ((x.astype(np.uint32) << 21) |
              (y.astype(np.uint32) << 10) |
              amplitudes.astype(np.uint32))
    return packed_data

def encode_frame(array, timestamp, threshold):
    """Apply zero suppression and encode a full frame."""
    x, y, amplitudes = zero_suppression(array, threshold)
    packed_data = encode_hits(x, y, amplitudes)
    header = np.array([(MAGIC_WORD, timestamp, len(amplitudes))], dtype=frame_header_dtype)
    return header, packed_data

def make_frame_header(timestamp, nhits):
    """Create a frame header."""
    header = np.array([(MAGIC_WORD, timestamp, nhits)], dtype=frame_header_dtype)
    return header

def make_file_header(frame_width, frame_height):
    """Create a file header."""
    header = np.array([(MAGIC_WORD, frame_width, frame_height)], dtype=file_header_dtype)
    return header

def decode_hits(packed_data):
    """Decode hits from the compact 32-bit representation."""
    x = (packed_data >> 21) & 0x7FF
    y = (packed_data >> 10) & 0x7FF
    amplitudes = packed_data & 0x3FF
    return x, y, amplitudes

def decode_header(header):
    """Decode frame header."""
    magic_word = header['magic_word']
    if magic_word != MAGIC_WORD:
        raise ValueError("Invalid magic word in frame header")
    timestamp = header['timestamp']
    nhits = header['nhits']
    return timestamp, nhits

def decode_file_header(header):
    """Decode file header."""
    magic_word = header['magic_word']
    if magic_word != MAGIC_WORD:
        raise ValueError("Invalid magic word in file header")
    frame_width = header['frame_width']
    frame_height = header['frame_height']
    return frame_width, frame_height

def read_file_header(infile):
    """Read and decode the file header from the input file."""
    header_bytes = infile.read(file_header_dtype.itemsize)
    if len(header_bytes) < file_header_dtype.itemsize:
        raise EOFError("Unexpected end of file while reading file header")
    header = np.frombuffer(header_bytes, dtype=file_header_dtype)[0]
    return decode_file_header(header)

def read_and_decode_frame(infile):
    """Read and decode a single frame from the input file."""
    header_bytes = infile.read(frame_header_dtype.itemsize)
    if len(header_bytes) < frame_header_dtype.itemsize:
        raise EOFError("Unexpected end of file while reading frame header") 
    header = np.frombuffer(header_bytes, dtype=frame_header_dtype)[0]
    timestamp, nhits = decode_header(header)
    packed_data_bytes = infile.read(nhits * 4)
    if len(packed_data_bytes) < nhits * 4:
        raise EOFError("Unexpected end of file while reading frame data")
    packed_data = np.frombuffer(packed_data_bytes, dtype=np.uint32)
    x, y, amplitudes = decode_hits(packed_data)
    return timestamp, x, y, amplitudes

