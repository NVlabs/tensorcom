# Tensorcom

Tensorcom is a way of loading training data into deep learning frameworks
quickly and portably. You can write a single data loading/augmentation
pipeline and train one or more jobs in the same or different frameworks
with it.

Both Keras and PyTorch can use the Python `Connection` object for input,
but MessagePack and ZMQ libraries exist in all major languages, making
it easy to write servers and input operators for any framework.

Tensorcom replaces the use of `multiprocessing` in Python for that purpose.
Both use separate processes for loading and augmentation, but by making
the processes and communications explicit, you gain some significant advantages:

 - the same augmentation pipeline can be used with different DL frameworks
 - augmentation processes can easily be run on multiple machines
 - output from a single automentation pipeline can be shared by many training jobs
 - you can start up and test the augmentation pipeline before you start the Dl jobs
 - DL frameworks wanting to use `tensorcom` only need a small library to handle input

Using `tensorcom` for training is very simple. First, start up a data server;
for Imagenet, there are two example jobs. The `serve-imagenet-dir` program
illustrates how to use the standard PyTorch Imagenet `DataLoader` to serve
training data:

```
    $ serve-imagenet-dir -d /data/imagenet -b 64 zpub://127.0.0.1:7880
```

The server will give you information about the rate at which it serves image batches.
Your training loop then becomes very simple:

```
    training = tensorcom.Connection("zsub://127.0.0.1:7880", epoch=1000000)
    for xs, ys in training:
        train_batch(xs, ys)
```

If you want multiple jobs for augmentation, just use more publishers using
Bash-style brace notation: `zpub://127.0.0.1:788{0..3}` and `zsub://127.0.0.1:788{0..3}`.

Note that you can start up multiple training jobs connecting to the same server.

# Command Line Tools

There are some command line programs to help with developing and debugging these
jobs:

 - tensormon -- connect to a data server and monitor throughput
 - tensorshow -- show images from input batches
 - tensorstat -- compute statistics over input data samples

# Examples

 - serve-imagenet-dir -- serve Imagenet data from a file system using PyTorch
 - serve-imagenet-shards -- serve Imagenet from shards using `webloader`
 - keras.ipynb -- simple example of using Keras with tensorcom
 - pytorch.ipynb -- simple example of using PyTorch with tensorcom

# ZMQ URLs

There is no official standard for ZMQ URLs. This library uses the following notation:

Socket types:

 - zpush / zpull -- standard PUSH/PULL sockets
 - zrpush / zrpull -- reverse PUSH/PULL connections (PUSH socket is server / PULL socket connects)
 - zpub / zsub -- standard PUB/SUB sockets
 - zrpub / zrsub -- reverse PUB/SUB connections

 The pub/sub servers allow the same augmentation pipeline to be shared by multiple
 learning jobs.

 Default transport is TCP/IP, but you can choose IPC as in `zpush+ipc://mypath`.

# Connection Objects

The major way of interacting with the library is through the `Connection` object.
It simply gives you an iterator over training samples.

# Encodings

Data is encoded in a simple binary tensor format; see `codec.py` for details.
The same format can also be used for saving and loading lists of
tensors from disk (extension: `.ten`).
Data is encoded on 64 byte aligned boundaries to allow easy memory
mapping and direct use by CPUs and GPUs.
