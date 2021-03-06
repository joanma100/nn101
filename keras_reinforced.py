from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys



def train_human(s1):
	l1_size = 100
	l2_size = 100
	score = s1
	notes = set(s1)

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
	model.add(Dropout(0.2))
	model.add(LSTM(l2_size, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(len(notes)))
	model.add(Activation('softmax'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop')


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
		model.fit(x, y, batch_size=128, nb_epoch=10)

		start_index = random.randint(0, len(score) - maxlen - 1)

		for diversity in [1.]:
			print()
			print('----- diversity:', diversity)

			generated = ''
			#phrase = score[start_index: start_index + maxlen]
			phrase = s1[0]		
			generated += phrase
			print('----- Generating with seed: "' + phrase + '"')
			sys.stdout.write(generated+' ')

			for i in range(400):
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
	return model



def evaluate_human():
	return



def reinforced():
	return


if __name__ == "__main__":
	
	text = 'abababababab'*100;
	
	# Train N2 from real text: scores text
	n1 = train_human(text)

	# Training N1
	for iteration in range(100):
		# 1) generate output from N1
		# 2) evaluate N1 output with N2
		# 3) select best pool
		# 4) train N1

