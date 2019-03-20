import numpy as np


def read_from_data(filename):
    lines = []
    with open(filename, 'r') as fd:
        lines = fd.read().splitlines()
    return lines


def get_encodings(type='train'):
    data = np.array(read_from_data('../data/qa.894.raw.{}.txt'.format(type)))

    # select questions (odd rows)
    questions = data[::2]

    # define vocabulary
    voc = dict()
    indices = dict()
    cnt = 0

    for q in questions:
        tokens = q.split()
        for token in tokens:
            voc[token] = voc.get(token, 0) + 1
            if token not in indices:
                indices[token] = cnt
                cnt += 1

    voc_size = len(voc)

    # build one-hot encodings
    sentences = []

    for q in questions:
        sent = []
        tokens = q.split()

        for token in tokens:
            one_hot_encoding = np.zeros(voc_size)
            index = indices.get(token, -1)
            one_hot_encoding[index] = 1
            sent.append(one_hot_encoding)

        for i in range(30-len(tokens)):
            if len(sent) > 30:
                print("?")
            if len(sent) >= 30:
                break
            sent.append(np.zeros(voc_size))

        if len(sent) == 30:
            sentences.append(sent)

    return np.array(sentences)


if __name__ == '__main__':
    encodings = get_encodings()

    print(len(encodings))
