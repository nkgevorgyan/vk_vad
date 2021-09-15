# VK Voice Activity Detection

## To run the model 

```
net = torch.load_net(os.getcwd() + '/models/' + title + '_epoch' + str(10).zfill(3) + '.net')
data = h5py_cache.File('data/data.hdf5', 'r')
```

## Evaluation on my test set 

1. Accuracy: **0.814**
2. F1-score: **0.768**
3. ROC-AUC : **0.864**

## Threshold values

1.  FA = 1% for **t = 0.0006**
2.  FR = 1% for **t = 0.9999**
3.  FA = FR for **t = 0.0659**

## Evaluation on dev set

1. Accuracy: **0.893**
2. F1-score: **0.943**
3. ROC-AUC : **0.587**
