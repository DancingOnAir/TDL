import sys
import os

print(os.pardir)
sys.path.append(os.pardir)

import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi


def test_ppmi():
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)
    W = ppmi(C)

    np.set_printoptions(precision=3)
    print('covariance matrix')
    print(C)
    print('-' * 50)
    print('PPMI')
    print(W)


if __name__ == '__main__':
    test_ppmi()