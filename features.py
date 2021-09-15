import python_speech_features
import array
import h5py_cache
import glob
import re
from pydub import AudioSegment
import soundfile as sf
import h5py
import numpy as np
import webrtcvad
import os
from preprocessing import DataMerger

# Specify the desired WAV-format.
SAMPLE_RATE = 16000
SAMPLE_CHANNELS = 1
SAMPLE_WIDTH = 2

# Name of folder to save the data files in.
DATA_FOLDER = 'data'

# Min/max length for slicing the voice files.
SLICE_MIN_MS = 1000
SLICE_MAX_MS = 5000

# Frame size to use for the labelling.
FRAME_SIZE_MS = 10

# Convert slice ms to frame size.
SLICE_MIN = int(SLICE_MIN_MS / FRAME_SIZE_MS)
SLICE_MAX = int(SLICE_MAX_MS / FRAME_SIZE_MS)

# Calculate frame size in data points.
FRAME_SIZE = int(SAMPLE_RATE * (FRAME_SIZE_MS / 1000.0))

speech_dataset = DataMerger('speech', 'LibriSpeech')
noise_dataset = DataMerger('noise', 'QUT-NOISE')

speech_data = speech_dataset.data
noise_data = noise_dataset.data


data = h5py_cache.File(DATA_FOLDER + '/data.hdf5', 'a', chunk_cache_mem_size=1024 ** 3)
noise_levels_db = {'None': None, '-15': -15, '-3': -3}

mfcc_window_frame_size = 4

#np.random.seed(1337)


def add_noise(speech_frames, noise_frames, align_frames, noise_level_db):
    # Convert to tracks.
    speech_track = (AudioSegment(data=array.array('h', speech_frames.flatten()),
                                 sample_width=SAMPLE_WIDTH, frame_rate=SAMPLE_RATE,
                                 channels=SAMPLE_CHANNELS))

    noise_track = (AudioSegment(data=array.array('h', noise_frames.flatten()),
                                sample_width=SAMPLE_WIDTH, frame_rate=SAMPLE_RATE,
                                channels=SAMPLE_CHANNELS))

    # Overlay noise.
    track = noise_track.overlay(speech_track, gain_during_overlay=noise_level_db)

    # Get frames data from track.
    raw = np.array(track.get_array_of_samples(), dtype=np.int16)
    frames = np.array(np.split(raw, len(raw) / FRAME_SIZE))

    # Add previous frames to align MFCC window.
    frames_aligned = np.concatenate((align_frames, frames))

    mfcc = python_speech_features.mfcc(frames_aligned, SAMPLE_RATE, winstep=(FRAME_SIZE_MS / 1000),
                                       winlen=mfcc_window_frame_size * (FRAME_SIZE_MS / 1000), nfft=2048)

    # First MFCC feature is just the DC offset.
    mfcc = mfcc[:, 1:]
    delta = python_speech_features.delta(mfcc, 2)

    return frames, mfcc, delta


if 'labels' not in data:

    print('Shuffling speech data and randomly adding 50% silence.')

    pos = 0
    l = len(speech_dataset.data['frames'])
    slices = []

    # Split speech data randomly within the given slice length.
    while pos + SLICE_MIN < l:
        slice_indexing = (pos, pos + np.random.randint(SLICE_MIN, SLICE_MAX + 1))
        slices.append(slice_indexing)
        pos = slice_indexing[1]

    # Add remainder to last slice.
    slices[-1] = (slices[-1][0], l)

    pos = 0

    # Add random silence (50%) to the track within the given slice length.
    while pos + SLICE_MIN < l:
        length = np.random.randint(SLICE_MIN, SLICE_MAX + 1)
        slice_indexing = (length, length)
        slices.append(slice_indexing)
        pos += length

    # Get total frame count.
    total = l + pos + mfcc_window_frame_size

    # Shuffle the content randomly.
    np.random.shuffle(slices)

    # Create data set for input.
    for key in noise_levels_db:
        data.create_dataset('frames-' + key, (total, FRAME_SIZE), dtype=np.dtype(np.int16))
        data.create_dataset('mfcc-' + key, (total, 12), dtype=np.dtype(np.float32))
        data.create_dataset('delta-' + key, (total, 12), dtype=np.dtype(np.float32))

    # Create data set for labels.
    dt = np.dtype(np.int8)
    data.create_dataset('labels', (total,), dtype=dt)

    pos = 0

    # Construct speech data.
    for s in slices:

        # Silence?
        if s[0] == s[1]:
            frames = np.zeros((s[0], FRAME_SIZE))
            labels = np.zeros(s[0])
        # Otherwise use speech data.
        else:
            frames = speech_data['frames'][s[0]: s[1]]
            labels = speech_data['labels'][s[0]: s[1]]

        # Pick random noise to add.
        i = np.random.randint(0, len(noise_data['frames']) - len(labels))
        noise = noise_data['frames'][i: i + len(labels)]

        # Setup noise levels.
        for key in noise_levels_db:

            # Get previous frames to align MFCC window with new data.
            if pos == 0:
                align_frames = np.zeros((mfcc_window_frame_size - 1, FRAME_SIZE))
            else:
                align_frames = data['frames-' + key][pos - mfcc_window_frame_size + 1: pos]

            # Add noise and get frames, MFCC and delta of MFCC.
            frames, mfcc, delta = add_noise(np.int16(frames), np.int16(noise),
                                            np.int16(align_frames), noise_levels_db[key])

            data['frames-' + key][pos: pos + len(labels)] = frames
            data['mfcc-' + key][pos: pos + len(labels)] = mfcc
            data['delta-' + key][pos: pos + len(labels)] = delta

        # Add labels.
        data['labels'][pos: pos + len(labels)] = labels

        pos += len(labels)
        print('Generating data ({0:.2f} %)'.format((pos * 100) / total), end='\r', flush=True)

    data.flush()

    print('\nDone!')

else:
    print('Speech data already generated. Skipping.')