from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import argparse


def convert(sentence):
    sentence = sentence.split()
    sentence = sentence[:-4]
    sentence.append("?")
    sentence = ' '.join(sentence)
    return sentence


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words=10):
    result = list()
    in_text = seed_text
    # generate a maximum of 10 words if no <END> is predicted
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre')
        # predict probabilities for each word
        predicted_class = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        prpedicted_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_class:
                prpedicted_word = word
                break

        # append to input
        in_text += ' ' + prpedicted_word
        result.append(prpedicted_word)
        if prpedicted_word == 'end':
            break
    sentence = ' '.join(result)
    return sentence


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to model")
    ap.add_argument("-t", "--tokenizer", required=True,
                    help="path to tokenizer")
    args = vars(ap.parse_args())

    MODEL_NAME = args['model']
    TOKENIZER_NAME = args['tokenizer']

    model = load_model('./trained_model/' + MODEL_NAME)

    # load the tokenizer
    tokenizer = load(open('./trained_model/' + TOKENIZER_NAME, 'rb'))

    print("Give a sentence to make a prediction ... (e.g.  \'How many chairs are in the image1 ?\')\n")
    sentence = input("Sentence: ")
    # load the model

    check = sentence.split()
    if check[-3] == "the" and check[-4] == "in":
        sentence = convert(sentence)

    print(sentence)

    # generate new text
    generated = generate_seq(model, tokenizer, 29, sentence.lower())
    print(generated)
