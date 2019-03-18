from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import data

print(len(data.train_x))

def start():
    input_dim = 30

    # this is the size of our encoded representations
    encoding_dim = 25  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

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

    train_x = data.train_x

    print(train_x.shape)

    validate_x = train_x[5000:]
    train_x = train_x[:5000]

    autoencoder.fit(train_x, train_x,
                    epochs=500,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(validate_x, validate_x)
    )

if __name__ == '__main__':
    start()

