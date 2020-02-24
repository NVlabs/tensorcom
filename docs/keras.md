# Simple Example of using WebLoader with Keras


```python
!test -f training.tgz || curl http://storage.googleapis.com/lpr-imagenet-augmented/imagenet_train-0000-000.tgz -o training.tgz
!test -f testing.tgz || curl http://storage.googleapis.com/lpr-imagenet-augmented/imagenet_val-0000.tgz -o testing.tgz
```

In a separate window, start the server with:

```
        serve-imagenet -u training.tgz -b 64
```


```python
from importlib import reload
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import models, layers
import os
import numpy as np
import tensorcom as tc
from webdataset import WebDataset
from webdataset.loader import WebLoader
```


```python
mean = np.array([0.485, 0.456, 0.406], "f")
std = np.array([0.229, 0.224, 0.225], "f")
def norm_image(xs):
    return (xs-mean[None,None,:])/std[None,None,:]
def norm_cls(ys):
    return keras.utils.to_categorical(ys-1, 1000)
```

We're using the regular DataLoader for the test data set. In fact, it's best just to download the test data set.


```python
#urls = "http://storage.googleapis.com/lpr-imagenet-augmented/imagenet_val-0000.tgz"
urls = "testing.tgz"
test_batch_size = 256
test_num_batches = 50000//test_batch_size
testing_ds = WebDataset(urls, extensions="ppm;png;jpg cls", transforms=[norm_image, norm_cls])
testing = WebLoader(testing_ds, batch_size=test_batch_size)
xs, ys = next(iter(testing))
print(xs.shape, xs.dtype, np.amin(xs), np.amax(xs))
print(ys.shape, ys.dtype, np.amin(ys), np.amax(ys))
```

The `Connection` class is similar to the `WebLoader` class, but the decompression and batching happens in the `tensorcom` server process.


```python
#urls = "http://storage.googleapis.com/lpr-imagenet-augmented/imagenet_train-{0000..0147}-{000..019}.tgz"
num_samples = 1000000
training = tc.Connection("zsub://127.0.0.1:7880",
                         epoch=num_samples,
                         batch_transforms=[norm_image, norm_cls])
xs, ys = next(iter(training))
print(xs.shape, xs.dtype, np.amin(xs), np.amax(xs))
print(ys.shape, ys.dtype, np.amin(ys), np.amax(ys))
```


```python
batch_norm = True
dropout = 0.5
nclasses = 1000
shape = (224, 224, 3)
batchsize = 32
lr = 0.001 # was: 0.0001

def vgg_block(d, r=2, name=None, size=(3, 3), mp_size=(2, 2), mp_strides=(2, 2)):
    result = []
    for i in range(r):
        name1 = name + "-{}".format(i) if name is not None else None
        if batch_norm:
            result += [
                layers.Conv2D(d, size, padding="same", name=name1),
                layers.BatchNormalization(),
                layers.Activation("relu")
            ]
        else:
            result += [layers.Conv2D(d, size, activation="relu", padding="same", name=name1)]        
    name1 = name + "-mp" if name is not None else None
    result += [layers.MaxPooling2D(mp_size, mp_strides, name=name1)]
    return result

model = models.Sequential(
    [layers.Lambda(lambda x: x + 0, input_shape=shape)] +         
    vgg_block(64, 2) + 
    vgg_block(128, 2) +
    vgg_block(256, 3) +
    vgg_block(512, 3) +
    vgg_block(512, 3) +
    [layers.Flatten(name="flatten"),
     layers.Dense(4096, activation="relu", name="fc1"),
     layers.Dropout(rate=1-dropout),
     layers.Dense(4096, activation="relu", name="fc2"),
     layers.Dropout(rate=1-dropout),
     layers.Dense(nclasses, activation="softmax", name="prediction")])

opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit_generator(iter(training),
              epochs=1,
              steps_per_epoch=1000, #num_batches,
              validation_data=iter(testing),
              validation_steps=100, #test_num_batches,
              shuffle=True)
```
