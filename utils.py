import python_speech_features
import array
import h5py_cache

data = h5py_cache.File(DATA_FOLDER + '/data.hdf5', 'a', chunk_cache_mem_size=1024 ** 3)
noise_levels_db = {'None': None, '-15': -15, '-3': -3}

mfcc_window_frame_size = 4

speech_data = speech_dataset.data
noise_data = noise_dataset.data

np.random.seed(1337)


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


class Vis:

    @staticmethod
    def _norm_raw(raw):
        '''
        Private function.
        Normalize the raw signal into a [0..1] range.
        '''
        return raw / np.max(np.abs(raw), axis=0)

    @staticmethod
    def _time_axis(raw, labels):
        '''
        Private function.
        Generates time axis for a raw signal and its labels.
        '''
        time = np.linspace(0, len(raw) / SAMPLE_RATE, num=len(raw))
        time_labels = np.linspace(0, len(raw) / SAMPLE_RATE, num=len(labels))
        return time, time_labels

    @staticmethod
    def _plot_waveform(frames, labels, title='Sample'):
        '''
        Private function.
        Plot a raw signal as waveform and its corresponding labels.
        '''
        raw = Vis._norm_raw(frames.flatten())
        time, time_labels = Vis._time_axis(raw, labels)

        plt.figure(1, figsize=(16, 3))
        plt.title(title)
        plt.plot(time, raw)
        plt.plot(time_labels, labels - 0.5)
        plt.show()

    @staticmethod
    def plot_sample(frames, labels, title='Sample', show_distribution=True):
        '''
        Plot a sample with its original labels
        (before noise is applied to sample).
        '''
        Vis._plot_waveform(frames, labels, title)

        # Print label distribution if enabled.
        if show_distribution:
            voice = (labels.tolist().count(1) * 100) / len(labels)
            silence = (labels.tolist().count(0) * 100) / len(labels)
            print('{0:.0f} % voice {1:.0f} % silence'.format(voice, silence))

    @staticmethod
    def plot_sample_webrtc(frames, sensitivity=0):
        '''
        Plot a sample labeled with WebRTC VAD
        (after noise is applied to sample).
        Sensitivity is an integer from 0 to 2,
        with 0 being the most sensitive.
        '''
        vad = webrtcvad.Vad(sensitivity)
        labels = np.array([1 if vad.is_speech(f.tobytes(), sample_rate=SAMPLE_RATE) else 0 for f in frames])
        Vis._plot_waveform(frames, labels, title='Sample (WebRTC)')

    @staticmethod
    def plot_features(mfcc=None, delta=None):
        '''
        Plots the MFCC and delta-features
        for a given sample.
        '''
        if mfcc is not None:
            plt.figure(1, figsize=(16, 3))
            plt.plot(mfcc)
            plt.title('MFCC ({0} features)'.format(mfcc.shape[1]))
            plt.show()

        if delta is not None:
            plt.figure(1, figsize=(16, 3))
            plt.plot(delta)
            plt.title('Deltas ({0} features)'.format(mfcc.shape[1]))
            plt.show()