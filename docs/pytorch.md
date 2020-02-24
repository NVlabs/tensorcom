# Simple Example of using Tensorcom with PyTorch

In a separate window, start the server with:

```
        curl http://storage.googleapis.com/lpr-imagenet-augmented/imagenet_train-0000-000.tgz > training.tgz
        serve-imagenet -u training.tgz -b 64
```

Or, if you have a local clone of the entire dataset:

```
        serve-imagenet -u http://my-server/some-path/imagenet_train-{0000..0147}-{000..019}.tgz
```


```python
%matplotlib inline
```


```python
from importlib import reload

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torchvision import models

from webdataset import WebDataset
from torch.utils.data import DataLoader
import tensorcom as tc

from IPython import display
import matplotlib.pyplot as plt

```


```python
mean = np.array([0.485, 0.456, 0.406], "f")
std = np.array([0.229, 0.224, 0.225], "f")
def norm_image(xs):
    return (xs-mean[None,None,:])/std[None,None,:]
def norm_cls(ys):
    ys = ys.astype(np.int64)
    return ys-1
```


```python
#urls = "http://storage.googleapis.com/lpr-imagenet-augmented/imagenet_train-{0000..0147}-{000..019}.tgz"
num_samples = 1000000
training = tc.Connection("zsub://127.0.0.1:7880",
                         epoch=num_samples,
                         batch_transforms=[norm_image, norm_cls],
                         converters="torch")
xs, ys = next(iter(training))
print(xs.shape, xs.dtype, xs.min().item(), xs.max().item())
print(ys.shape, ys.dtype, ys.min().item(), ys.max().item())
```

    torch.Size([32, 3, 224, 224]) torch.float32 -2.1179039478302 1136.357177734375
    torch.Size([32]) torch.int64 33 998



```python
model = models.resnet50()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.0001, momentum=0.9, weight_decay=0.0)
```


```python
def rtplot(ys, xs=None, sigma=20, fig=None):
    fig = fig or plt.gcf()
    fig.clf()
    fig.add_subplot(1, 1, 1)
    ax = fig.get_axes()[0]
    ax.cla()
    from scipy.ndimage import filters
    ys = filters.gaussian_filter(np.array(ys, "f"), sigma, mode="nearest")
    if xs is not None:
        plt.plot(xs, ys)
    else:
        plt.plot(ys)
    display.clear_output(wait=True)
    display.display(fig)
        
def train_for(steps, losses=[]):
    model.cuda()
    model.train()
    for i, (input, target) in enumerate(training):
        if i>=steps: break
        output = model(input.cuda())
        loss = criterion(output, target.cuda())
        losses.append(loss.item())
        rtplot(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    display.clear_output(wait=True)
        
train_for(10000)
```


![png](pytorch_files/pytorch_7_0.png)



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-6-f38f70836786> in <module>
         28     display.clear_output(wait=True)
         29 
    ---> 30 train_for(10000)
    

    <ipython-input-6-f38f70836786> in train_for(steps, losses)
         17     model.cuda()
         18     model.train()
    ---> 19     for i, (input, target) in enumerate(training):
         20         if i>=steps: break
         21         output = model(input.cuda())


    ~/exp/tensorcom/tensorcom/zcom.py in items(self, report)
        366             if self.total > 0 and self.count >= self.total:
        367                 return
    --> 368             result = self.recv()
        369             if report > 0 and count >= next_report:
        370                 print("count", count, self.stats.summary())


    ~/exp/tensorcom/tensorcom/zcom.py in recv(self)
        327             self.socket.recv_multipart()
        328         if self.multipart:
    --> 329             data = tenbin.zrecv_multipart(self.socket, self.infos)
        330         else:
        331             data = tenbin.zrecv_single(self.socket, self.infos)


    ~/exp/tensorcom/tensorcom/tenbin.py in zrecv_multipart(socket, infos)
        222 def zrecv_multipart(socket, infos=False):
        223     """Receive arrays as a multipart ZMQ message."""
    --> 224     return decode_list(socket.recv_multipart(), infos=infos)
        225 
        226 def sctp_send(socket, dest, l, infos=None):


    /usr/lib/python3/dist-packages/zmq/sugar/socket.py in recv_multipart(self, flags, copy, track)
        465             for any of the reasons :func:`~Socket.recv` might fail
        466         """
    --> 467         parts = [self.recv(flags, copy=copy, track=track)]
        468         # have first part already, only loop while more to receive
        469         while self.getsockopt(zmq.RCVMORE):


    zmq/backend/cython/socket.pyx in zmq.backend.cython.socket.Socket.recv()


    zmq/backend/cython/socket.pyx in zmq.backend.cython.socket.Socket.recv()


    zmq/backend/cython/socket.pyx in zmq.backend.cython.socket._recv_copy()


    /usr/lib/python3/dist-packages/zmq/backend/cython/checkrc.pxd in zmq.backend.cython.checkrc._check_rc()


    KeyboardInterrupt: 



![png](pytorch_files/pytorch_7_2.png)



```python

```
