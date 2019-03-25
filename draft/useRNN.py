from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import argparse
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')

def convert(sentence):
    sentence = sentence.split()
    sentence = sentence[:-4]
    sentence.append("?")
    sentence = ' '.join(sentence)
    return sentence


def wups(ground_truth, predicted, threshold=0.9):
    ground_truth_synset = wn.synset('{}.n.01'.format(ground_truth))
    predicted_synset = wn.synset('{}.n.01'.format(predicted))

    wup = ground_truth_synset.wup_similarity(predicted_synset)

    return wup if wup >= threshold else wup * 0.1


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
    #sentence = ' '.join(result)
    return result
    #return sentence


def read_data(filename):
    import numpy as np
    lines = []
    with open(filename, 'r') as fd:
        lines = fd.read().splitlines()

    return np.array(lines)

def evaluate(model, tokenizer, n=10):
    from random import randint
    data = read_data('../data/qa.894.raw.test.txt')

    questions = data[::2]
    answers = data[1::2]

    print(questions)
    print(answers)

    indices = []

    total = 0
    current_test = 1

    while current_test <= n:
        idx = randint(0, len(questions))
        while idx in indices:
            idx = randint(0, len(questions))

        indices.append(idx)

        sentence = questions[idx].strip().lower()
        ground_answer = answers[idx].split()[0].strip().lower()

        print('Testing:', sentence)
        print('Ground answer:', ground_answer)

        check = sentence.split()
        if check[-3] == "the" and check[-4] == "in":
            sentence = convert(sentence)

        generated = generate_seq(model, tokenizer, 29, sentence.lower())
        print(generated)

        if len(generated) > 0:
            print('Predicted answer:', generated[0])
            try:
                score = wups(ground_answer, generated[0])
                print('WUPS:', score)
                total += score
                current_test += 1
            except:
                print("unable to calculate WUPS score")

    avg = total / max(n, 1)

    print('Average WUPS over {} predictions:'.format(n), avg)

    return avg


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False, default="model_120_1553439784.6067197.h5",
                    help="path to model")
    ap.add_argument("-t", "--tokenizer", required=False, default="tokenizer_120_1553439784.8818188.pkl",
                    help="path to tokenizer")
    ap.add_argument("-w", "--wups", action='store_true', required=False,
                    help="path to tokenizer")
    args = vars(ap.parse_args())

    MODEL_NAME = args['model']
    TOKENIZER_NAME = args['tokenizer']
    EVAL_WUPS = args['wups']


    model = load_model('./trained_model/' + MODEL_NAME)

    # load the tokenizer
    tokenizer = load(open('./trained_model/' + TOKENIZER_NAME, 'rb'))

    if not EVAL_WUPS:
        print("Give a sentence to make a prediction ... (e.g.  \'How many chairs are in the image1 ?\')\n")
        sentence = input("Sentence: ")
        ground_answer = input("Ground truth answer: ")

        check = sentence.split()
        if check[-3] == "the" and check[-4] == "in":
            sentence = convert(sentence)

        print(sentence)

        # generate new text
        generated = generate_seq(model, tokenizer, 29, sentence.lower())
        print(generated)

        if len(generated) > 0:
            try:
                score = wups(ground_answer, generated[0])
                print('WUPS:', score)
            except:
                print("unable to calculate WUPS score")
    else:
        print('WUPS evaluation...')
        avg_wups = evaluate(model, tokenizer, 2000)
