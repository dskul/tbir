from numpy import array
from numpy import asarray
from numpy import zeros
import numpy as np
import string
import copy
from numpy import array
from pickle import dump
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

from utils import read_from_data

def test():
    # define documents
    docs = ['Well done!',
    		'Good work',
    		'Great effort',
    		'nice work',
    		'Excellent!',
    		'Weak',
    		'Poor effort!',
    		'not good',
    		'poor work',
    		'Could have done better.']
    # define class labels
    labels = array([1,1,1,1,1,0,0,0,0,0])


    vocab_size = 50
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    print(encoded_docs)


    # pad documents to a max length of 4 words
    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)

    # define the model
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))


def clean_file():
    type = 'train'
    data = read_from_data('../data/qa.894.raw.{}.txt'.format(type))
    trainset = []
    for i in range(0,len(data),2):
        data[i] = data[i].split()
        #data[i] = data[i][:-4]
        #data[i].append("?")
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
    save_doc(trainset,"../data/testset.txt")



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
    # load
    in_filename = '../data/testset.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')

    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    print(sequences[:3])
    max_length = len(max(sequences,key=len))
    print(max_length)
    sequences = array(sequences)
    sequences = pad_sequences(sequences, maxlen=max_length+1, padding='pre')
    # vocabulary size
    print(sequences[:3])
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)
    X, y = sequences[:,:-1], sequences[:,-1]
    print(len(X))
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    # fit model

    model.fit(X,y, batch_size=128, epochs=100, validation_split=0.2)

    # save the model to file
    model.save('model.h5')
    # save the tokenizer
    dump(tokenizer, open('tokenizer.pkl', 'wb'))




if __name__ == '__main__':
    run()
