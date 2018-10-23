"""
Authors:    Dario Cazzani
"""
#!/usr/bin/python
from __future__ import division
import tensorflow as tf
import numpy as np

def log10(x):
    num = tf.log(x)
    den = tf.log(tf.constant(10, dtype=num.dtype))
    return(tf.div(num, den))

def overlapping_slicer_3D(_input, block_size, stride):
    _input_rank = int(len(_input.get_shape()))
    blocks = []
    n = _input.get_shape().as_list()[_input_rank-1]
    low = range(0, n, stride)
    high = range(block_size, n+1, stride)
    low_high = zip(low, high)
    for low, high in low_high:
	blocks.append(_input[:, low:high])
    return(tf.stack(blocks, _input_rank-1))

def angle(z):
    if z.dtype == tf.complex128:
        dtype = tf.float64
    elif z.dtype == tf.complex64:
        dtype = tf.float32
    else:
        raise ValueError('input z must be of type complex64 or complex128')

    x = tf.real(z)
    y = tf.imag(z)
    x_neg = tf.cast(x < 0.0, dtype)
    y_neg = tf.cast(y < 0.0, dtype)
    y_pos = tf.cast(y >= 0.0, dtype)
    offset = x_neg * (y_pos - y_neg) * np.pi
    return tf.atan(y / x) + offset

def is_power2(x):
    return x > 0 and ((x & (x - 1)) == 0)
