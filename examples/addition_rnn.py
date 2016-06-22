# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs

'''

from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from six.moves import range


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        #print ("self.chars : ", self.chars)
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        #for w,v in self.char_indices.items():
        #    print ("w=  ", w)
        #    print ("v=  ", v)
        #print ("self.char_indices : ", self.char_indices)
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        #print ("self.indices_char : ", self.indices_char)
        self.maxlen = maxlen
        #print ("self.maxlen : ", self.maxlen)

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            #print("i = ", i)
            #print("c = ", c)
            #print("char_indices =  ", self.char_indices[c])
            X[i, self.char_indices[c]] = 1
            #for j in range(0,len(X[i])):
            #    print("X[i][j] = " , X[i][j])
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 50000
DIGITS = 15 
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = DIGITS + 1 + DIGITS

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

#chars = '0123456789+ '
chars = vocab_hash.keys()
#for i in range(0, len(chars)):
    #print("c = ", chars[i])
#print("chars = ",chars)
ctable = CharacterTable(chars, MAXLEN)
print ("len of chars = ", len(chars))

seen = set()
print('Generating data...')
#while len(questions) < TRAINING_SIZE:
print ("len of questions = ", len(questions))
for i in range(0,len(questions)):
    query = questions[i]
    ans = anwsers[i]
    #for i in range(0,len(query)):
    #    print("word = ", query[i])
    #print("query = ", query)
    #print("anwser = ", ans)
print('Total addition questions:', len(questions))

print('Vectorization...')
X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
print("shape of X:", X.shape)
print("shape of y:", y.shape)
### SAM: X[i] and y[i] are both matrix, X[i] is the expresstion of the formular like : 43+234, and y[i] is the calculate outcome like : 234.
for i, sentence in enumerate(questions):
    #print("X:", sentence)
    #print("len of sentence:", len(sentence))
    X[i] = ctable.encode(sentence, maxlen=MAXLEN)
    #for j in range(0,len(X[i])):
    #    print("j = ", X[i][j])
    #print("X:", X[i])
for i, sentence in enumerate(anwsers):
    #for l in range(0,len(sentence)):
    #    print("word = ",sentence[l])
    y[i] = ctable.encode(sentence, maxlen=MAXLEN)
    #for j in range(0,len(y[i])):
    #    print("j = ", y[i][j])
# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10
(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print ("----------1")
# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val))
    print ("----------2")
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
#    for i in range(10):
#        ind = np.random.randint(0, len(X_val))
#        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
#        preds = model.predict_classes(rowX, verbose=0)
#        q = ctable.decode(rowX[0])
#        correct = ctable.decode(rowy[0])
#        guess = ctable.decode(preds[0], calc_argmax=False)
#        print('Q', q[::-1] if INVERT else q)
#        print('T', correct)
#        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
#        print('---')
