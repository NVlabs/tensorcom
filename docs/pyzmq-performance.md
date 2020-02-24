```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
import numpy as np
import zmq
import subprocess
from subprocess import Popen
```


```python
data = np.ones((256, 3, 224, 224), dtype=np.float16)
bin = data.tobytes()
```


```python
%%writefile server.py
import numpy as np
import zmq
data = np.ones((256, 3, 224, 224), dtype=np.float16)
bindata = data.tobytes()
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:7988")
for i in range(10000):
    socket.send(bindata)
```

    Overwriting server.py



```python
proc = Popen(["python3", "server.py"])
proc.poll()
```


```python
%%time
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://127.0.0.1:7988")
total = 0
for i in range(100):
    data = socket.recv()
    total += len(data)
    if i%10==0: print(i, len(data))
print(total/1e6)
del socket
proc.terminate()
proc.wait()
```

    0 77070336
    10 77070336
    20 77070336
    30 77070336
    40 77070336
    50 77070336
    60 77070336
    70 77070336
    80 77070336
    90 77070336
    7707.0336
    CPU times: user 1.3 s, sys: 8.64 s, total: 9.94 s
    Wall time: 6.13 s



```python
proc = Popen(["python3", "server.py"])
proc.poll()
```


```python
%%time
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://127.0.0.1:7988")
total = 0
for i in range(100):
    data = socket.recv()
    a = np.frombuffer(data, dtype=np.float16).reshape(256, 3, 224, 224)
    if i%10==0: print(i, a.size)
    total += a.size
print(total/1e6)
del socket
proc.terminate()
proc.wait()
```

    0 38535168
    10 38535168
    20 38535168
    30 38535168
    40 38535168
    50 38535168
    60 38535168
    70 38535168
    80 38535168
    90 38535168
    3853.5168
    CPU times: user 1.36 s, sys: 8.76 s, total: 10.1 s
    Wall time: 6.38 s



```python
proc = Popen(["python3", "server.py"])
proc.poll()
```
