# Tensorcom Data Server

This illustrates transmitting a training set via tensorcom. Here, we use a standard Torch `Dataloader` as a data source.


```python
import sys
import torch
from torchvision import datasets, transforms
import numpy as np
import tensorcom
```


```python
loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=32, shuffle=True)
```

We use simple URLs with `zpub`, `zsub`, `zpush`, and `zpull` schemas for making ZMQ connections. There are also reverse versions `zr...`, which reverse the connect/bind schemes.

Here we use a ZMQ PUB socket for distributing data. Such a socket will send data asynchronously, whether clients are connected or not.


```python
serve = tensorcom.Connection()
serve.connect("zpub://127.0.0.1:7888")
```

In this sample library, all tensors are represented as NumPy arrays, so we have to convert the PyTorch tensors to NumPy before sending.

For many application, sending floating point data in `float16` format is sufficient and potentially faster when networking is involved.


```python
for epoch in range(5):
    sys.stderr.write("{} ".format(epoch))
    for i, (xs, ys) in enumerate(loader):
        xs = np.array(xs).astype('float16')
        ys = np.array(ys).astype('int32')
        serve.send([xs, ys])
```

    0 1 2 3 4 

Note that to achieve higher data rates, you can start up multiple publishers and then connect to them from a single training job.

Note also that, under the covers, PyTorch's parallel `Dataloader` functions very similarly to this approach; it also uses multiple processes and IPC for loading data asynchronously. However, by making the communication explicit with Tensorcom, we can use the same preprocessing pipelines for PyTorch and TensorFlow, and we can also share training data between multiple jobs.

Also note that you can use any data loading and augmentation framework you like in the sender, and combine it with any DL framework. In particular, you can use PyTorch `Dataset`/`DataLoader`, you can use TensorFlow input pipelines, and you can use the `dlinputs` framework.


```python

```
