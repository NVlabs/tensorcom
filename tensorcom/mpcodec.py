#!/usr/bin/env python
# coding: utf-8

# NB: this file is obsolete; do not use functions in this file

import msgpack
import numpy as np

# The following are small helper functions (`np2dict`, `dict2np`) that recursively convert back and forth between dictionaries and NumPy arrays by recognizing dictionaries containing special keys. These functions can be used with any JSON-style representation, including JSON and msgpack. They are intended for use with msgpack.
#
# Data can either be encoded inline or in a separate parts list. When encoded in a separate parts list, binary array data can be sent in a multipart ZMQ message, which allows more efficient transmission, more efficient memory management, and easier msgpack decoding (since the msgpack representation becomes very small).
#
# While these functions in Python allow encoding of arbitrary JSON-like data structures, for `tensorcom` and typical input pipelines, it is sufficient to just handle the special case of list-of-values or dictionary-of-values, where a value can be a number, a string, or a tensor. Coding these two special cases may be significantly easier in languages like C++.



allowable_dtypes64 = {np.dtype(x) for x in {"uint8", "uint16", "int16", "int32", "int64",
                                            "float16", "float32", "float64"}}
allowable_dtypes = {np.dtype(x) for x in {"uint8", "uint16", "int16", "int32",
                                          "float16", "float32"}}


def check_acceptable_input_type(a, allow64=False):
    """Checks whether `a` is acceptable for input queues.
    
    Inputs to DL pipelines should be tuples of tensors/scalars.

    :param a: sample data
    :param allow64: allow 64 bit data types (Default value = False)

    """
    assert isinstance(a, (list, tuple)), a
    for x in a:
        if "torch.Tensor" in str(type(x)):
            continue
        assert isinstance(x, (int, float, str, bytes, np.ndarray)), (a, x)
        if isinstance(x, np.ndarray):
            if allow64:
                assert x.dtype in allowable_dtypes64, x.dtype
            else:
                assert x.dtype in allowable_dtypes, x.dtype


def np2dict(a, parts=None):
    """Recursively convert numpy tensors in data structures to dictionaries.

    :param a: a Python data structure containing numpy arrays
    :param parts: encode using parts if list (Default value = None)

    """
    if isinstance(a, np.ndarray):
        assert a.dtype in allowable_dtypes64, a.dtype
        if parts is None:
            return dict(_shape=list(a.shape),
                        _dtype=str(a.dtype),
                        _data=a.tobytes())
        else:
            index = len(parts)
            parts.append(a.tobytes())
            return dict(_shape=list(a.shape),
                        _dtype=str(a.dtype),
                        _part=index)
    elif isinstance(a, list):
        return [np2dict(x, parts) for x in a]
    elif isinstance(a, dict):
        return {k: np2dict(v, parts) for k, v in a.items()}
    else:
        return a


def getdata(a, parts):
    """Get the data portion of a tensor.

    :param a: data dictionary
    :param parts: parts list

    Data can be encoded in several different ways:

    - _data -- inline as a string
    - _part -- reference to the parts list

    Optional and unimplemented:

    - _jpeg -- JPEG image encoding
    - _png -- PNG image encoding
    """
    if "_part" in a:
        return parts[a["_part"]]
    elif "_data" in a:
        return a["_data"]
    # Transmission of tensors in compressed image form is useful in some
    # circumstances where bandwidth is more constrained than compute.
    # While JPEG and PNG in the wild can yield any shape (h, w, d) for d=1...4,
    # as part of tensorcom, we require that the output of the JPEG/PNG decoder
    # matches the given shape exactly (no implicit grayscale conversions, etc.).
    # We can impose that requirement because the JPEG/PNG images encoded here
    # are already the output of preprocessing/augmentation.
    # Extension: _jpeg and _png can be lists or listss of lists, in which
    # case the output ends up being the same as calling array([[image, ...], ...])
    # Extension: eventually, we may also want to support "_mjpeg" and "_mp4" as
    # compression formats
    # Extension: support 16 bit versions of JPEG and PNG
    elif "_jpeg" in a:
        raise Exception("unimplemented")
        # decode JPEG; if float output required, divide by 255.0
    elif "_png" in a:
        raise Exception("unimplemented")
        # decode PNG; if float output required, divide by 255.0
    else:
        raise Exception("cannot find tensor data")


def keys_as_strings(a):
    """Return the set of keys as ASCII encoded strings.

    :param a: dictionary

    """
    return {k.decode("ascii") if isinstance(k, bytes) else k: v for k, v in a.items()}


def dict2np(a, parts=None, device=None):
    """Recursively convert dictionaries to numpy tensors.

    :param a: Python data structure.
    :param parts: parts list for _part (Default value = None)
    :param device: device placement for PyTorch (None means NumPy) (Default value = None)

    """
    if isinstance(a, dict):
        a = keys_as_strings(a)
        if "_shape" in a and "_dtype" in a:
            # TODO: We should probably check for unrecognized keywords here
            data = getdata(a, parts)
            data = np.frombuffer(data, dtype=a["_dtype"]).reshape(a["_shape"])
            # The following three transformations are useful to allow compact
            # storage of tensors in uint8, int16, and float16 formats while still
            # giving full dynamic range.
            if "_logscale" in a:
                # this allows storage of log numbers; scaling is needed for uint8 formats
                b, m = a["_logscale"]
                data = np.exp(data * float(m) + float(b))
            if "_scale" in a:
                # this allows rescaling of uint8 and float16 formats, whose range is somewhat limited
                b, m = a["_scale"]
                data = data * float(m) + float(b)
            if "_astype" in a:
                # this allows storage in a different format from what the input pipeline needs
                data = data.astype(a["_astype"])
            if device is not None:
                if device=="torch":
                    device = "cpu"
                import torch
                data = torch.as_tensor(data).to(device)
            return data
        else:
            return {k: dict2np(v, parts, device) for k, v in a.items()}
    elif isinstance(a, list):
        return [dict2np(x, parts, device) for x in a]
    else:
        return a
