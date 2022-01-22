from text_preprocess import preprocess
import numpy as np


def create_co_matrix(corpus, vocabulary_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocabulary_size, vocabulary_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx, right_idx = idx - i, idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
                pass
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def test_co_matrix():
    text = "You say goodbye and I say hello."

    corpus, word_to_id, id_to_word = preprocess(text)
    co_matrix = create_co_matrix(corpus, len(word_to_id))
    print(co_matrix)
    print(type(co_matrix[0, 0]))


if __name__ == '__main__':
    test_co_matrix()
