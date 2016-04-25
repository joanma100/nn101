from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical


model = Sequential()
model.add(Dense(3, input_dim=8, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(8, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adadelta', loss='categorical_crossentropy')#, metrics=['accuracy'])

X_train = to_categorical(range(0,8),8)

model.fit(X_train, X_train, nb_epoch=1000, batch_size=8)

print model.predict(X_train, batch_size=16, verbose=1)

from keras.utils.visualize_util import plot
plot(model, to_file='model.png')

