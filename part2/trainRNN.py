import copy
from numpy import array
from pickle import dump
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding, Dropout
import time
import json


def read_from_data(filename):
    lines = []
    with open(filename, 'r') as fd:
        lines = fd.read().splitlines()
    return lines


def clean_file():
    type = 'train'
    data = read_from_data('../data/qa.894.raw.{}.txt'.format(type))
    trainset = []
    for i in range(0,len(data),2):
        data[i] = data[i].split()
        if i == 0:
            print(data[i])
        data[i+1] = data[i+1].split()
        data[i+1].append("<END>")
        for res in data[i+1]:
            if i == 0:
                print(res)
            data[i].append(res)
            trainset.append(copy.deepcopy(data[i]))
            print(data[i])
    save_doc(trainset,"../data/testsetWithImages.txt")


# save tokens to file, one dialog per line
def save_doc(lines, filename):
    seq = []
    for line in lines:
        data = ' '.join(line)
        seq.append(data)
    result = '\n'.join(seq)
    file = open(filename, 'w')
    file.write(result)
    file.close()


def load_doc(filename):
    # open the file as read only
	file = open(filename, 'r')
    # read all text
	text = file.read()
    # close the file
	file.close()
	return text


def run():
    # load sequences
    in_filename = '../data/testset.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')

    #load json image features
    with open("../data/img_features.json") as file:
        imageFeatures = json.load(file)
    print(len(imageFeatures["image1"]))


    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    max_length = len(max(sequences,key=len))

    sequences = array(sequences)
    sequences = pad_sequences(sequences, maxlen=max_length+1, padding='pre')

    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)
    input, y = sequences[:,:-1], sequences[:,-1]

    y = to_categorical(y, num_classes=vocab_size)
    seq_length = input.shape[1]

    # define model
    EPOCHS = 120
    BATCH_SIZE = 256
    VALIDATION_SPLIT = 0.2
    DROPOUT_RATE_1 = 0.4
    DROPOUT_RATE_2 = 0.3
    DROPOUT_RATE_3 = 0.25
    LATENT_DIM     = 100

    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(LATENT_DIM, return_sequences=True))
    # model.add(Dropout(DROPOUT_RATE_1))
    model.add(LSTM(LATENT_DIM))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(DROPOUT_RATE_2))
    model.add(Dense(vocab_size, activation='softmax'))
    # model.add(Dropout(DROPOUT_RATE_3))
    print(model.summary())
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    # fit model

    model.fit(input,y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)

    # save the model to file
    model.save('./trained_model/model_{epochs}_{id}.h5'.format(epochs=EPOCHS, id=time.time()))
    # save the tokenizer
    dump(tokenizer, open('./trained_model/tokenizer_{epochs}_{id}.pkl'.format(epochs=EPOCHS, id=time.time()), 'wb'))


if __name__ == '__main__':
    run()
