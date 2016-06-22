'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = './x'
FIN = file(path, 'r')
vocab_hash  =  {}
questions = []
anwsers = []
num = 0
for line in FIN.readlines():
    line = line.strip()
    arr = line.split("#_#")
    if len(arr) < 2:
        continue
    question = arr[0]
    anwser = arr[1]
    arr = question.split(" ")
    for i in range(0, len(arr)):
        if(arr[i] == ""):
            continue
        if arr[i] not in vocab_hash:
            vocab_hash[arr[i]] = num;
            num = num + 1
    questions.append(arr)
    arr = anwser.split(" ")
    for i in range(0, len(arr)):
         if arr[i] not in vocab_hash:
            vocab_hash[arr[i]] = num;
            num = num + 1
    anwsers.append(arr)
#char_indices = dict((c, i) for i, c in enumerate(chars))
char_indices = vocab_hash
indices_char = dict((i, c) for i, c in vocab_hash.items())

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
print('Vectorization...')
X = np.zeros((len(questions), maxlen, len(vocab_hash)), dtype=np.bool)
y = np.zeros((len(anwsers), len(vocab_hash)), dtype=np.bool)
for i, sentence in enumerate(questions):
    #print("i = ",i)						### the i'th sentence.
    #print("sentence = ",sentence)			### content of i'th sentence.
    for t, char in enumerate(sentence):
        print("t = ",t)					### the t'th char of sentence.
        print("char = ",char)				### content of t'th char.
        X[i, t, vocab_hash[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
exit()

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
