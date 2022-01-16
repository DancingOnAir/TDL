from text_preprocess import preprocess
from create_co_matrix import create_co_matrix
import numpy as np


def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)

    return np.dot(nx, ny)


def test_cos_similarity():
    text = "You say goodbye and I say hello."

    corpus, word_to_id, id_to_word = preprocess(text)
    co_matrix = create_co_matrix(corpus, len(word_to_id))

    c0 = co_matrix[word_to_id['you']]
    c1 = co_matrix[word_to_id['i']]
    print(cos_similarity(c0, c1))


if __name__ == '__main__':
    test_cos_similarity()


