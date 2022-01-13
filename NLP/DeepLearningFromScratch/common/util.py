import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = 1 + int(H + 2 * pad - filter_h) // stride
    out_w = 1 + int(W + 2 * pad - filter_w) // stride

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + out_h * stride
        for x in range(filter_w):
            x_max = x + out_w * stride
            # https://blog.csdn.net/cjbct/article/details/100749205
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # https://zhuanlan.zhihu.com/p/72381219
    # reshape default order = 'C',
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = 1 + (H + 2 * pad - filter_h) // stride
    out_w = 1 + (W + 2 * pad - filter_w) // stride
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    return img