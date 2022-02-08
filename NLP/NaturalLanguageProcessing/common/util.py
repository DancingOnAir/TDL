import numpy as np


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = dict()
    id_to_word = dict()

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            id_to_word[new_id] = word
            word_to_id[word] = new_id

    corpus = np.array([word_to_id[word] for word in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)

    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('[query] %s' % query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(word_to_id)
    similarities = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarities[i] = cos_similarity(query_vec, word_matrix[i])

    count = 0
    for i in (-1 * similarities).argsort():
        if id_to_word[i] == query:
            continue
        print("%s: %s" % (id_to_word[i], similarities[i]))
        count += 1
        if count >= top:
            return


def ppmi(co_matrix, verbose=False, eps=1e-8):
    M = np.zeros_like(co_matrix, dtype=np.float32)
    N = np.sum(co_matrix)
    S = np.sum(co_matrix, axis=0)
    total = co_matrix.shape[0] * co_matrix.shape[1]
    cnt = 0

    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            pmi = np.log2(co_matrix[i, j] * N / (S[i] * S[j]) + eps)
            M[i, j] = max(0, pmi)

        if verbose:
            cnt += 1
            if cnt % (total // 100 + 1) == 0:
                print("%.lf%% done" % (100 * cnt / total))
    return M


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size: -window_size]
    contexts = list()

    for i in range(window_size, len(corpus) - window_size):
        cs = list()
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[i + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot