import numpy as np


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = dict()
    id_to_word = dict()
    for w in words:
        if w not in word_to_id:
            new_id = len(id_to_word)
            id_to_word[new_id] = w
            word_to_id[w] = new_id

    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def test_text_preprocess():
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)

    print('corpus: ' + str(corpus))


if __name__ == '__main__':
    test_text_preprocess()
