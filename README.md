# VK Voice Activity Detection

## How to run the model on the existing wav files

1. Prepare the data (making frames and getting features)

```
test_prov = preprocessing.DataMerger('test', 'for_devs')
test_prov.prepare_files()
test_prov.collect_frames()
test_prov.label_frames()
```
2. Specify parameters of the model
```
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
```
3. Create h5py to keep the data in.
```
test_data = h5py_cache.File(DATA_FOLDER + '/test_data.hdf5', 'a', chunk_cache_mem_size=1024 ** 3)
mfcc_window_frame_size = 4
l = len(test_prov.data['frames'])
total = l  + mfcc_window_frame_size

test_data.create_dataset('frames', (total, FRAME_SIZE), dtype=np.dtype(np.int16))
test_data.create_dataset('mfcc', (total, 12), dtype=np.dtype(np.float32))
test_data.create_dataset('delta', (total, 12), dtype=np.dtype(np.float32))
test_data.create_dataset('labels', (total,), dtype=np.dtype(np.int8))

number_frames = len(test_prov.data['frames'])
for i in range(number_frames):
    print("frame {} out of {}".format(i, number_frames))
    mfcc = python_speech_features.mfcc(test_prov.data['frames'][i], SAMPLE_RATE, winstep=(FRAME_SIZE_MS / 1000),
                                       winlen=mfcc_window_frame_size * (FRAME_SIZE_MS / 1000), nfft=2048)
    mfcc = mfcc[:, 1:]
    delta = python_speech_features.delta(mfcc, 2)
    test_data['frames'][i] = test_prov.data['frames'][i]
    test_data['mfcc'][i] = mfcc
    test_data['delta'][i] = mfcc
    test_data['labels'][i] = test_prov.data['labels'][i]
```
    
4. Dataset object for iteration
```    
class Dataset_test(torch.utils.data.Dataset):
    
    def __init__(self, data, data_start, data_size, batch_size, step_size, frame_count, noise='None'):

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

        mfcc = self.data['mfcc'][curr_idx: curr_idx + end_idx]
        delta = self.data['delta'][curr_idx: curr_idx + end_idx]
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

        # Load data and get label
        #X = torch.from_numpy(np.array(x)).float().cpu()
        #y = torch.from_numpy(np.array(y)).cpu()

        return x, y
        
```

5. Upload the data and run through the network
```
data = h5py_cache.File('data/test_data.hdf5', 'r')

test_data = Dataset_test(data, 0, 1042189, frame_count=3, 
                    step_size=1, batch_size=BATCH_SIZE, noise='None')
                    
net.eval()

y_true, y_score = [], []

for i in range(test_data.batch_count):

    X, y = test_data[i]
    X = Variable(torch.from_numpy(np.array(X)).float())
    y = Variable(torch.from_numpy(np.array(y))).long()

    if torch.cuda.is_available():
        X = X.cuda()

    out = net(X)

    if torch.cuda.is_available():
        out = out.cpu()
        y = y.cpu()

    # Add true labels.
    y_true.extend(y.data.numpy())

    y_score = [1 if y >= t else 0 for idx, y in enumerate(out.data.numpy()[:, 1])]
    
```                    

## To train the model

1. Create **data** folder and locate all the training data there
2. Preprocess this data

```    
python preprocessing.py
        
```
3. Add noise to data and create features for the training 

```    
python features.py
        
```

4. Train the model

```    
python train.py
        
```

## Architecture 

Simple model with LSTM and two additional Linear Layers was used.

For the evaluation metrics was used **accuracy** at first. 

But since we have imballanced classes case, I decided to use also F1-score and ROC-AUC.

                    
## Evaluation on my test set 

1. Accuracy: **0.814**
2. F1-score: **0.768**
3. ROC-AUC : **0.864**

## Threshold values

1.  FA = 1% for **t = 0.0006**
2.  FR = 1% for **t = 0.9999**
3.  FA = FR for **t = 0.0659**

## Evaluation on dev set

Dev set after preprocessing: https://drive.google.com/file/d/1RO6xVUODr38EfRwrtLI_b_SQvFjLrNkY/view?usp=sharing

1. Accuracy: **0.893**
2. F1-score: **0.943**
3. ROC-AUC : **0.587**
