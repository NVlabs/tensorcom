import numpy as np
from tensorcom import zcom
import random


def test_zmq():
    port = random.randint(17000, 38999)
    source = zcom.Connection(f"zpush://127.0.0.1:{port}")
    sink = zcom.Connection(f"zpull://127.0.0.1:{port}")
    a = np.random.uniform(size=(9, 7)).astype("float16")
    source.send([a, a])
    b, c = sink.recv()
    del sink
    del source
    assert (a == b).all()
    assert (a == c).all()
