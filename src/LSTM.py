import preprocessing as preproc
import helpers
from tqdm import tqdm
import time
import text_representation as text_repr
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Embedding, Activation, Flatten, GlobalMaxPooling1D, LSTM
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


start_time = time.time()

full_dataset = False
glove_pretrained = True

train_data, y, test_data = helpers.get_processed_data(full_dataset=full_dataset)
num_words = text_repr.define_num_words(train_data, full_dataset)
word_embeddings = dict()
if glove_pretrained:
    word_embeddings = text_repr.loadGloveModel("../data/twitter-datasets/glove.twitter.27B.200d.txt")


tokenizer = text_repr.fit_tokenizer(train_data, num_words)
embedding_matrix = text_repr.get_embedding_matrix(tokenizer, word_embeddings, num_words)

print('embedding_matrix.shape', embedding_matrix.shape)

tweet_max_length = text_repr.define_tweet_max_len(train_data, full_dataset)
tokenizer = text_repr.fit_tokenizer(train_data, num_words)
train_data_seq = tokenizer.texts_to_sequences(train_data)
test_data_seq = tokenizer.texts_to_sequences(test_data)
train_data_seq = pad_sequences(train_data_seq, maxlen=tweet_max_length)
test_data_seq = pad_sequences(test_data_seq, maxlen=tweet_max_length)

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=200, weights=[embedding_matrix], 
                    input_length=tweet_max_length, trainable=True))
model.add(LSTM(units=32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(units=32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train, X_test, y_train, y_test = train_test_split(train_data_seq, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=1)

