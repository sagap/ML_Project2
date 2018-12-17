import preprocessing as preproc
import helpers
from tqdm import tqdm
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Embedding, Activation, Flatten, GlobalMaxPooling1D
from keras.optimizers import Adam, SGD
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

start_time = time.time()
full_dataset = True
train_data, y, test_data = helpers.get_processed_data(full_dataset=full_dataset)

vectorizer = CountVectorizer(min_df=2)
count_vect = vectorizer.fit_transform(train_data)
vocabulary = vectorizer.get_feature_names()
no_vocabulary = len(vocabulary)
print('num of words: ', no_vocabulary)

num_words = 120000
sum_words = count_vect.sum(axis=0)
word_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
word_freq_sorted = sorted(word_freq, key = lambda x : x[1], reverse=True)
word_sorted = [x[0] for x in word_freq_sorted]
print('first 5: ', word_sorted[:5])

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_data)
sequences = tokenizer.texts_to_sequences(train_data)
length = []
for x in train_data:
    length.append(len(x.split()))
print('max sentence length:', max(length))
tweet_max_length = 130

train_data_seq = pad_sequences(sequences, maxlen=tweet_max_length)

model_cnn_02 = Sequential()
e = Embedding(num_words, 200, input_length=tweet_max_length)
model_cnn_02.add(e)
model_cnn_02.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_02.add(GlobalMaxPooling1D())
model_cnn_02.add(Dense(256, activation='relu'))
model_cnn_02.add(Dense(1, activation='sigmoid'))
model_cnn_02.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(train_data_seq, y, test_size=0.15, random_state=42)

model_cnn_02.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=1)