import torch
import h5py_cache
import numpy as np

NOISE_LEVELS = ['None', '-15', '-3']
BATCH_SIZE = 2048
FRAMES = 3
FEATURES = 24
STEP_SIZE = 1


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, data_start, data_size, batch_size, step_size, frame_count, noise='None'):
        'Initialization'
        self.data = data
        self.data_start = data_start
        self.data_size = data_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.frame_count = frame_count
        self.noise_level = noise

        n = int((self.data_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data['labels'])

  def __getitem__(self, idx):
        'Generates one sample of data'
        # Select sample
        curr_idx = self.data_start + idx * self.batch_size * self.step_size
        end_idx = self.frame_count + self.step_size * self.batch_size
        
        mfcc = self.data['mfcc-' + self.noise_level][curr_idx: curr_idx + end_idx]
        delta = self.data['delta-' + self.noise_level][curr_idx: curr_idx + end_idx]
        labels = self.data['labels'][curr_idx: curr_idx + end_idx]

        x, y, i = [], [], 0

        # Get batches
        while len(y) < self.batch_size:
            # Get data for the window.
            X = np.hstack((mfcc[i: i + self.frame_count], delta[i: i + self.frame_count]))

            # Append sequence to list of frames
            x.append(X)

            # Select label from center of sequence as label for that sequence.
            y_range = labels[i: i + self.frame_count]
            y.append(int(y_range[int(self.frame_count / 2)]))

            # Increment window using set step size
            i += self.step_size

        return x, y


def get_dataset(noise='None', val=0.1, test=0.1):
    DATA_FOLDER = 'data'
    data = h5py_cache.File(DATA_FOLDER + '/data.hdf5', 'r')
    dataset_size = len(data['labels'])

    train_index = 0
    val_index = int((1.0 - val - test) * dataset_size)
    test_index = int((1.0 - test) * dataset_size)
       
    train_size = val_index
    val_size = test_index - val_index
    test_size = dataset_size - test_index
    train_list, val_list, test_list = [], [], []

    for noise in NOISE_LEVELS:
        train_list.append(Dataset(data, train_index, train_size, frame_count=3, step_size=1, batch_size=BATCH_SIZE, noise=noise))
        val_list.append(Dataset(data, val_index, val_size, frame_count=3, step_size=1, batch_size=BATCH_SIZE, noise=noise))
        test_list.append(Dataset(data, test_index, test_size, frame_count=3, step_size=1, batch_size=BATCH_SIZE, noise=noise))

    return train_list, val_list, test_list



