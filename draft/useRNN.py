import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

import sys
import copy
import numpy as np
from pickle import load

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
sys.stderr = stderr

import argparse
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet', quiet=True)
import globals

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
    for it in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre')
        # predict probabilities for each word
        predicted_class = model.predict_classes(encoded, verbose=0)
        # get 2nd most likely answer
        y_prob = model.predict(encoded)[0].tolist()
        encoded_copy = copy.deepcopy(y_prob)
        encoded_copy.sort()
        predicted_class2 = y_prob.index(encoded_copy[-2])
        # map predicted word index to word
        predicted_word = None
        predicted_word2 = None
        start_end = False
        for word, index in tokenizer.word_index.items():
            if index == predicted_class:
                predicted_word = word
                if predicted_word == "end" and it == 0:
                    start_end = True
            if index == predicted_class2:
                predicted_word2 = word

        if start_end == True:
            predicted_word = predicted_word2
        # append to input
        in_text += ' ' + predicted_word
        result.append(predicted_word)
        if predicted_word == 'end':
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

    indices = []

    total = 0
    total_tests = n
    current_test = 1
    progress = 0

    while current_test <= n:
        idx = randint(0, len(questions))
        while idx in indices or idx >= len(questions):
            idx = randint(0, len(questions) - 1)

        indices.append(idx)

        sentence = questions[idx].strip().lower()
        ground_answer = answers[idx].split()[0].strip().lower()

        if globals.DEBUG_MODE:
            print('Testing:', sentence)
            print('Ground answer:', ground_answer)

        check = sentence.split()
        if check[-3] == "the" and check[-4] == "in":
            sentence = convert(sentence)

        generated = generate_seq(model, tokenizer, 29, sentence.lower())

        if globals.DEBUG_MODE:
            print(generated)

        if len(generated) > 0:
            if globals.DEBUG_MODE:
                print('Predicted answer:', generated[0])
            try:
                score = wups(ground_answer, generated[0])
                if globals.DEBUG_MODE:
                    print('WUPS:', score)
                total += score
                current_test += 1
            except:
                if globals.DEBUG_MODE:
                    print("unable to calculate WUPS score")

        if not globals.DEBUG_MODE:
            progress = ((current_test * 1.0) / total_tests) * 100
            progress = 100 if progress > 100 else progress
            sys.stdout.write("Progress: %0.2f%%   \r" % progress)
            sys.stdout.flush()

    if not globals.DEBUG_MODE:
        print()

    print()

    avg = total / max(n, 1)

    print('###############################################')
    print('# Average WUPS over {} predictions: %0.2f'.format(n) % avg)
    print('###############################################')
    print()

    return avg


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False, default="model_120_1553439784.6067197.h5",
                    help="path to model")
    ap.add_argument("-t", "--tokenizer", required=False, default="tokenizer_120_1553439784.8818188.pkl",
                    help="path to tokenizer")
    ap.add_argument("-wc", "--wups-count", required=False, default=2000,
                    help="total number of question-answer test pairs to include in wups scoring")
    ap.add_argument("-w", "--wups", action='store_true', required=False,
                    help="include to start wups evaluation")
    ap.add_argument("-d", "--debug", action='store_true', required=False,
                    help="show debug information")
    args = vars(ap.parse_args())

    DISPLAY_HELP = args.get('help', False)
    MODEL_NAME = args['model']
    globals.DEBUG_MODE = args['debug']
    TOKENIZER_NAME = args['tokenizer']
    EVAL_WUPS = args['wups']
    WUPS_COUNT = int(args['wups_count'])

    if DISPLAY_HELP:
        ap.print_help()
        sys.exit(0)

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
        print('################################')
        print('# STARTING WUPS evaluation...')
        print('################################')
        print()
        avg_wups = evaluate(model, tokenizer, WUPS_COUNT)
