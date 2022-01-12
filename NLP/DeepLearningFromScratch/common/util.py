import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = 1 + int(H + 2 * pad - filter_h) // stride
    out_w = 1 + int(W + 2 * pad - filter_w) // stride

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # 这里6个参数的意义
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    return col
    pass


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = 1 + (H + 2 * pad - filter_h) // stride
    out_w = 1 + (W + 2 * pad - filter_w) // stride
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    return img