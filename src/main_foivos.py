import preprocessing as preproc
import helpers
from tqdm import tqdm
import time
import text_representation as text_repr

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Embedding, Activation, Flatten, GlobalMaxPooling1D
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
import cross_validation as cv


start_time = time.time()

full_dataset = True
NN = True
padding = True

train_data, y, test_data = helpers.get_processed_data(full_dataset=full_dataset)

num_words = text_repr.define_num_words(train_data, full_dataset)


if not NN:
    X_train, X_test = text_repr.get_features('glove', train_data, test_data, full_dataset, num_features=None)  
else:
    word_embeddings = text_repr.train_glove_WE(train_data, full_dataset, 
     num_features=200, predefined=True, build_embeddings=True)
    embedding_matrix, train_data_seq = text_repr.get_embedding_matrix(train_data, word_embeddings, num_words)


tweet_max_length = text_repr.define_tweet_max_len(train_data, full_dataset)
if padding:
    train_data_seq = pad_sequences(train_data_seq, maxlen=tweet_max_length)

elapsed_time = divmod(round((time.time() - start_time)), 60)
print('------\nElapsed time: {m} min {s} sec\n'.format(m=elapsed_time[0], s=elapsed_time[1]))


model_cnn_02 = Sequential()
e = Embedding(num_words, 200, weights=[embedding_matrix], input_length=tweet_max_length, trainable=True)
# e = Embedding(num_words, 200, input_length=tweet_max_length)
model_cnn_02.add(e)
model_cnn_02.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_02.add(GlobalMaxPooling1D())
model_cnn_02.add(Dense(256, activation='relu'))
model_cnn_02.add(Dense(1, activation='sigmoid'))
model_cnn_02.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

result = cv.cross_validation_NN(model_cnn_02, train_data_seq, y, num_of_iters)

# X_train, X_test, y_train, y_test = train_test_split(train_data_seq, y, test_size=0.20, random_state=42)

# model_cnn_02.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=1)

