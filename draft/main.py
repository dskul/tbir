from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import data
import utils
import random
#from testRNN import BlindRNN as BRNN



def start():
    input_dim = 1649

    # this is the size of our encoded representations
    encoding_dim = 68  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    #              Got same error with encoding_dim = 34!


    encoded_input = Input(shape=(encoding_dim,))


    # this is our input placeholder
    input_text = Input(shape=(input_dim,))

    print(input_text)

    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_text)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_text, decoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') #categorical_crossentropy

    encodings = utils.get_encodings()

    #train_x = data.train_x

    print(encodings.shape)

    validate_x = encodings[5000:]
    x_train = encodings[:5000]

    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(validate_x, validate_x)
    )

    print(autoencoder.summary())

    # Layer (type)                 Output Shape              Param #
    # =================================================================
    # input_1 (InputLayer)         (None, 1649)              0
    # _________________________________________________________________
    # dense_1 (Dense)              (None, 68)                112200
    # _________________________________________________________________
    # dense_2 (Dense)              (None, 1649)              113781
    # =================================================================
    # Total params: 225,981
    # Trainable params: 225,981
    # Non-trainable params: 0
    # _________________________________________________________________


    #encoding_layer = autoencoder.layers[-2]
    #encoder = Model(input_text, encoding_layer(input_text))
    decoder_layer = autoencoder.layers[-1]
    encoder = Model(input_text, encoded)

    encoded_texts = encoder.predict(encodings[0:10])
    print(encoded_texts)
    print(len(encoded_texts[0]))
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    print("-----------")

    # guides makes a vector for each input (sentence) and gives an resultlist

    keras_x = np.asarray([random.randint(0,10) for x in range(150)])
    print(keras_x,"\n",keras_x.shape)
    keras_x = keras_x.reshape(keras_x.shape[0],1)
    print(keras_x,"\n",keras_x.shape)







if __name__ == '__main__':
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))
    print(data.shape)
    print(labels.shape)
    start()
