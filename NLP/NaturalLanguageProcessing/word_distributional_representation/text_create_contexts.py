import sys
import os
sys.path.append(os.pardir)


from common.util import preprocess, create_contexts_target


def test_create_contexts_target():
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus)

    contexts, target = create_contexts_target(corpus)
    print(contexts)
    print(target)


if __name__ == '__main__':
    test_create_contexts_target()
