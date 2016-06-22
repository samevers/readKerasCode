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

chars = '0123456789+ '
ctable = CharacterTable(chars, MAXLEN)

questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    # Skip any addition questions we've already seen
    # Also skip any such that X+Y == Y+X (hence the sorting)
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1
    ans += ' ' * (DIGITS + 1 - len(ans))
    if INVERT:
        query = query[::-1]
    #print("type of query = ",type(query)) 		## str
    #print("type of ans = ",type(ans))			## str
    #print("query = ",query)
    #print("ans = ",ans)
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

print('Vectorization...')
X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
### SAM: X[i] and y[i] are both matrix, X[i] is the expresstion of the formular like : 43+234, and y[i] is the calculate outcome like : 234.
for i, sentence in enumerate(questions):
    #print("X:", sentence)							## 5+2
    #print("type of X:", type(sentence))			## str
    X[i] = ctable.encode(sentence, maxlen=MAXLEN)
for i, sentence in enumerate(expected):
    #print("sentence = ",sentence)
    y[i] = ctable.encode(sentence, maxlen=DIGITS + 1)
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
    '''
    Show the content and format of X_train,y_train.
	Both of X_train,y_train are 3d matrix, each sentence is expressed in a 2d matrix,
		that each row of the 2d matrix express a word 1-of-K vector.
		e.g.,
		 X_train =  [[[ True False False ..., False False False]
		              [ True False False ..., False False False]
		              [ True False False ..., False False False]
					  ...... ......
					]] 
		 y_train =  [[[ False False ..., False False False True]
		              [ True False False ..., False False False]
		              [ True False False ..., False False False]
					  ...... ......
					]] 
    '''
    #print("X_train = ",X_train)
    #print("y_train = ",y_train)
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
