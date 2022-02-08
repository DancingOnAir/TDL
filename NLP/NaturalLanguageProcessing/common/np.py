# coding: utf-8
from common.config import GPU


if GPU:
    import cupy as np

    print('-' * 60)
    print("cupy")
    print('-' * 60)

else:
    import numpy as np
