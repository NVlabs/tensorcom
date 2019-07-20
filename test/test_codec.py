import numpy as np
from tensorcom import zcom
from tensorcom import codec

def test_str64():
    s = "hello"
    assert s == codec.unstr64(codec.str64(s))

def test_header():
    a = np.random.uniform(size=(12, 19)).astype("f")
    shape, dtype, info = codec.decode_header(codec.encode_header(a, 'data'))
    assert a.shape == shape, (a.shape, shape)
    assert a.dtype == dtype, (a.dtype, dtype)
    assert info == "data"

def test_list():
    a = np.random.uniform(size=(12, 19)).astype("f")
    encoded = codec.encode_list([a,a], infos=["tensor1", "tensor2"])
    b, c = codec.decode_list(encoded, infos=["tensor1", "tensor2"])
    assert (a==b).all()
    assert (a==c).all()

def test_buffer():
    a = np.random.uniform(size=(1, 3, 4)).astype("float32")
    encoded = codec.encode_buffer([a, a], infos=["input", "target"])
    b, c = codec.decode_buffer(encoded, infos=["input", "target"])
    assert (a==b).all()
    assert (a==c).all()

def test_buffer():
    a = np.random.uniform(size=(7, 7)).astype("float16")
    codec.save("/tmp/_temp.ten", a, a, infos=["hello", "world"])
    b, c = codec.load("/tmp/_temp.ten", infos=["hello", "world"])
    assert (a==b).all()
    assert (a==c).all()

def test_zmq():
    import zmq
    import random
    port = random.randint(17000,18999)
    con = zmq.Context()
    sink = con.socket(zmq.PULL)
    sink.bind(f"tcp://127.0.0.1:{port}")
    source = con.socket(zmq.PUSH)
    source.connect(f"tcp://127.0.0.1:{port}")
    a = np.random.uniform(size=(7, 7)).astype("float16")
    codec.zsend_multipart(source, [a, a], infos=["za", "zb"])
    b, c = codec.zrecv_multipart(sink, infos=["za", "zb"])
    sink.close()
    source.close()
    del sink
    del source
    assert (a==b).all()
    assert (a==c).all()


def no_test_sctp():
    import socket
    import sctp
    import random
    port = random.randint(17000,18999)
    sk = sctp.sctpsocket_tcp(socket.AF_INET)
    sk.bind(("0.0.0.0", port))
    sk.listen(20)
    sr = sctp.sctpsocket_tcp(socket.AF_INET)
    a = np.random.uniform(size=(17)).astype("float16")
    codec.sctp_send(sk, ("127.0.0.1", port), [a, a])
    b, c = codec.sctp_recv(sr)
    sk.close()
    sr.close()
    del sk
    del sr
    assert (a==b).all()
    assert (a==c).all()
