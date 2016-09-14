from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


file_path = '../datasets/text/llull/obresdoctrinalis01llul_djvu.txt'
fh = open(file_path,"r")
s1 = fh.read()

l1_size = 512
l2_size = 250
l3_size = 50
l4_size = 512

score = s1
notes = set(s1)


print('Total chars:', len(notes))

notes_indices = dict((n, i) for i, n in enumerate(notes))
indices_notes = dict((i, n) for i, n in enumerate(notes))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 30
step = 3

bars = []
next_notes = []

for i in range(0, len(score) - maxlen, step):
	bars.append(score[i: i + maxlen])
	next_notes.append(score[i + maxlen])

print('nb sequences:', len(bars))

print('Vectorization...')
x = np.zeros((len(score), maxlen, len(notes)), dtype=np.bool)
y = np.zeros((len(score), len(notes)), dtype=np.bool)
for i, bars in enumerate(bars):
	for t, note in enumerate(bars):
		x[i, t, notes_indices[note]] = 1
		y[i, notes_indices[next_notes[i]]] = 1



# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(l1_size, return_sequences=True, input_shape=(maxlen, len(notes))))
model.add(Dropout(0.5))
#model.add(LSTM(l2_size, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(l3_size, return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(l4_size, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(len(notes)))
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
	print(np.shape(x), np.shape(y))
	model.fit(x, y, batch_size=128, nb_epoch=1)

	start_index = random.randint(0, len(score) - maxlen - 1)

	for diversity in [0.3, 0.5, 0.7, 1.]:
		print()
		print('----- diversity:', diversity)

		generated = ''
		phrase = score[start_index: start_index + maxlen]
		generated += phrase
		print('----- Generating with seed: "' + phrase + '"')
		sys.stdout.write(generated+' ')

		for i in range(300):
			xv = np.zeros((1, maxlen, len(notes)))
			for t, note in enumerate(phrase):
				xv[0, t, notes_indices[note]] = 1.

			preds = model.predict(xv, verbose=0)[0]
			next_index = sample(preds, diversity)
			next_note = indices_notes[next_index]

			generated += next_note
			phrase = phrase[1:] + next_note

			sys.stdout.write(next_note)
			sys.stdout.flush()
		print()
