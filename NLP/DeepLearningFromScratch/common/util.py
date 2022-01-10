def img2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = 1 + int(H + 2 * pad - filter_h) // stride
    out_w = 1 + int(W + 2 * pad - filter_w) // stride
    pass