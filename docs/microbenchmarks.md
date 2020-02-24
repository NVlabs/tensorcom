```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
import argparse
import multiprocessing
import sys
from functools import partial

import dlinputs as dli
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import simplejson
import torch
import torch.nn.functional as F
from dlinputs import filters, gopen, improc, utils
from matplotlib import cm
from torch import nn, optim
from torchvision import datasets, transforms
from itertools import islice
import time
from PIL import Image
import io

import tensorcom
```

# I/O


```python
%%time
total = 0
with open("./imagenet_train-0000.tar", "rb") as stream:
    while True:
        data = stream.read(1000000)
        if len(data)==0: break
        total += len(data)
print(total/1e6)
```

    1022.95552
    CPU times: user 0 ns, sys: 163 ms, total: 163 ms
    Wall time: 160 ms


# Tar Decoding and Decompressing


```python
%%time
total = 0
source = dli.gopen.open_source("./imagenet_train-0000.tar", decode=False)
for sample in islice(source, 0, 1000):
    total += len(sample["jpg"])
print(total/1e6)
```

    109.936435
    CPU times: user 262 ms, sys: 20.3 ms, total: 283 ms
    Wall time: 281 ms



```python
%%time
total = 0
source = dli.gopen.open_source("./imagenet_train-0000.tgz", decode=False)
for sample in islice(source, 0, 1000):
    total += len(sample["jpg"])
print(total/1e6)
```

    109.936435
    CPU times: user 394 ms, sys: 19.6 ms, total: 414 ms
    Wall time: 412 ms


# JPEG Decoding


```python
images = []
```


```python
%%time
total = 0
source = dli.gopen.open_source("./imagenet_train-0000.tgz", decode=dli.utils.autodecoder("PIL"))
for sample in islice(source, 0, 1000):
    total += sample["jpg"].size[0]
    images.append(sample["jpg"].convert("RGB"))
print(total)
```

    478969
    CPU times: user 8.17 s, sys: 305 ms, total: 8.48 s
    Wall time: 8.46 s


# JPEG Decoding with Multiprocessing


```python
%%writefile decoder.py
from PIL import Image
import io
def decode(sample):
    sample["jpg"] = Image.open(io.BytesIO(sample["jpg"]))
    return sample
```

    Overwriting decoder.py



```python
pool = multiprocessing.Pool(16)
from decoder import decode
```


```python
%%time
total = 0
source = dli.gopen.open_source("./imagenet_train-0000.tar", decode=False)
count = 0
for sample in pool.imap_unordered(decode, islice(source, 0, 10000)):
    total += sample["jpg"].size[0]
    count += 1
print(total, count)
```

    4727094 10000
    CPU times: user 11.9 s, sys: 3.91 s, total: 15.8 s
    Wall time: 16.2 s


# GPU-Based JPEG Decoding


```python
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import nvidia.dali.plugin.pytorch as dalipyt

with open("space.jpg", "rb") as stream:
    space = stream.read()
        
class DecoderPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, pipelined = False, async = False):
        super(DecoderPipe, self).__init__(batch_size,
                                         num_threads,
                                         device_id,
                                         exec_pipelined=pipelined,
                                         exec_async=async)
        self.input = ops.ExternalSource()
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)

    def define_graph(self):
        self.jpegs = self.input()
        return self.decode(self.jpegs)

pipe = DecoderPipe(1, 1, 0)
pipe.build()

def dali2torch(dali_tensor):
    import torch
    import ctypes
    assert dali_tensor.dtype() == "B"
    tensor = torch.zeros(dali_tensor.shape(), dtype=torch.uint8, device="cuda")
    assert dali_tensor.shape() == list(tensor.shape)
    dali_tensor.copy_to_external(ctypes.c_void_p(tensor.data_ptr()))
    return tensor

def gpudecode(jpeg):
    pipe.feed_input(pipe.jpegs, [jpeg])
    dali_tensors = pipe.run()[0]
    assert len(dali_tensors) == 1
    image = dali2torch(dali_tensors.at(0))
    return image
    
gpudecode(space).size()
```




    torch.Size([480, 640, 3])



## One-by-one Decoding


```python
%%time
total = 0
source = dli.gopen.open_source("./imagenet_train-0000.tgz", decode=False)
for sample in islice(source, 0, 1000):
    image = gpudecode(sample["jpg"])
    total += image.size(0)
print(total)
```

    405015
    CPU times: user 3.36 s, sys: 175 ms, total: 3.53 s
    Wall time: 3.53 s


## Multithreaded Batch Decoding


```python
pipe = DecoderPipe(1000, 8, 0)
pipe.build()
dali_tensors = None
```


```python
%%time
global dali_tensors
source = dli.gopen.open_source("./imagenet_train-0000.tgz", decode=False)
jpegs = [sample["jpg"] for sample in islice(source, 0, 1000)]
pipe.feed_input(pipe.jpegs, jpegs)
dali_tensors = pipe.run()[0]
images = [dali2torch(dali_tensors.at(i)) for i in range(len(dali_tensors))]
images = [a.cpu().numpy() for a in images]
total = np.sum([a.shape[0] for a in images])
print(total)
```

    405015
    CPU times: user 2.63 s, sys: 704 ms, total: 3.34 s
    Wall time: 1.38 s


# Data Augmentation


```python
%%time
augment = transforms.Compose([transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip()])
for image in images:
    augment(image)
```

    CPU times: user 943 ms, sys: 0 ns, total: 943 ms
    Wall time: 940 ms



```python
%%time
channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
augment = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])
for image in images:
    augment(image)
```

    CPU times: user 21.4 s, sys: 18 ms, total: 21.4 s
    Wall time: 8.18 s



```python
arrays = []
```


```python
%%time
augment = transforms.Compose([transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip()])
for image in images:
    a = np.asarray(augment(image)).astype(np.float16)
    arrays.append(a)
print(len(arrays))
```

    1000
    CPU times: user 1.65 s, sys: 67.7 ms, total: 1.72 s
    Wall time: 1.72 s


# Tensor Sending


```python
%%time
serve = tensorcom.Connection("zpub://127.0.0.1:7893")
for array in arrays:
    serve.send([array, 0])
serve.close()
print(array.shape)
```

    (224, 224, 3)
    CPU times: user 57.7 ms, sys: 8.22 ms, total: 65.9 ms
    Wall time: 64.2 ms


# Tensor Sending with Batching


```python
%%time
serve = tensorcom.Connection("zpub://127.0.0.1:7895")
for batch in filters.batched(50)(dict(img=array) for array in arrays):
    serve.send([batch["img"], 0])
serve.close()
print(batch["img"].shape)
```

    (50, 224, 224, 3)
    CPU times: user 137 ms, sys: 0 ns, total: 137 ms
    Wall time: 136 ms


# Summary

|Step              | s/1000 |
|------------------|--------|
|tar decoding      |  0.270 |
|tgz decoding      |  0.412 |
|JPEG decoding     |  8.640 |
|simple aug        |  0.920 |
|complex aug       |  7.920 |
|simple aug + conv |  1.760 |
|batch + send      |  0.158 |


```python

```
