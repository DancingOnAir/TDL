import os
import sys
sys.path.append(os.pardir)


from common.util import preprocess, create_co_matrix, most_similar


def test_most_similar():
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    word_matrix = create_co_matrix(corpus, vocab_size)

    most_similar('you', word_to_id, id_to_word, word_matrix, top=5)


if __name__ == '__main__':
    test_most_similar()
