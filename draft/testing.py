from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Embedding
from keras.models import Model, Sequential
import numpy as np
import data
import utils


# Prepare Glove File
def readGloveFile(gloveFile):
    with open(gloveFile, 'r') as f:
        wordToGlove = {}  # map from a token (word) to a Glove embedding vector
        wordToIndex = {}  # map from a token to an index
        indexToWord = {}  # map from an index to a token

        for line in f:
            record = line.strip().split()
            token = record[0] # take the token (word) from the text line
            wordToGlove[token] = np.array(record[1:], dtype=np.float64) # associate the Glove embedding vector to a that token (word)

        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  # 0 is reserved for masking in Keras (see above)
            wordToIndex[tok] = kerasIdx # associate an index to a token (word)
            indexToWord[kerasIdx] = tok # associate a word to a token (word). Note: inverse of dictionary above

    return wordToIndex, indexToWord, wordToGlove

# Create Pretrained Keras Embedding Layer
def createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, isTrainable):
    vocabLen = len(wordToIndex) + 1  # adding 1 to account for masking
    embDim = next(iter(wordToGlove.values())).shape[0]  # works with any glove dimensions (e.g. 50)

    embeddingMatrix = np.zeros((vocabLen, embDim))  # initialize with zeros
    for word, index in wordToIndex.items():
        embeddingMatrix[index, :] = wordToGlove[word] # create embedding: word index to Glove word embedding

    embeddingLayer = Embedding(vocabLen, embDim, weights=[embeddingMatrix], trainable=isTrainable)
    return embeddingLayer


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


from keras.preprocessing.text import Tokenizer

sentences_train = utils.read_from_data('../data/qa.894.raw.train.txt')

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(sentences_train[2])
print(X_train[2])

print(tokenizer.word_index)
print(len(tokenizer.word_index))

from keras.preprocessing.sequence import pad_sequences

max_length = max([len(l) for l in X_train])
print(max_length)

maxlen = 30

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

print(X_train[0, :])


embedding_dim = 30
embedding_matrix = create_embedding_matrix(
    '../data/glove/glove.6B.50d.txt',
    tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / vocab_size)


model = Sequential()
model.add(Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=False))
#model.add(GlobalMaxPool1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()





# usage
wordToIndex, indexToWord, wordToGlove = readGloveFile("../data/glove/glove.6B.50d.txt")
pretrainedEmbeddingLayer = createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, False)

input_layer = Input(shape=(30,), name='text_input')

embedded_text = pretrainedEmbeddingLayer(input_layer)

embedding_model = Model(inputs=input_layer, outputs=embedded_text)

x_train = data.train_x
print(x_train.shape)

input_sequence = x_train[0].reshape((1,30))
print(input_sequence.shape)
print(input_sequence)


# Extract the embeddings by calling the .predict() method
sequence_embeddings = embedding_model.predict(input_sequence)

print('embeddings:')
print(sequence_embeddings.shape)
print(sequence_embeddings)
