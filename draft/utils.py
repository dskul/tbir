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
    encodings = []

    for q in questions:
        tokens = q.split()
        one_hot_encoding = np.zeros(voc_size)

        for token in tokens:
            index = indices.get(token, -1)
            one_hot_encoding[index] = 1

        encodings.append(one_hot_encoding)

    return np.array(encodings)


if __name__ == '__main__':
    encodings = get_encodings()

    print(len(encodings))
