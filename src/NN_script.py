from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Embedding, Activation, Flatten
from keras.optimizers import Adam
import numpy as np
import re
import preprocessing as preproc
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import text_representation as repre

start_time = time.time()

preproc.do_preprocessing('../data/twitter-datasets/train_pos.txt')
preproc.do_preprocessing('../data/twitter-datasets/train_neg.txt')
print('Preprocessing finished...')

lines, y = preproc.return_processed_trainset_and_y(False)
print(len(lines))

vectorizer = repre.create_feature_representation('TfidfVectorizer', 1000)
X = vectorizer.fit_transform(lines)

adam = Adam(lr=0.02, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)

# create model
model = Sequential()
# model.add(Embedding(X.shape[0], 300, input_length=X.shape[1]))
# model.add(Flatten())
model.add(Dense(units=64, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

#compile model
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

#fit the model
model.fit(x=X, y=y, batch_size=32, epochs=100, verbose=1)

scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))