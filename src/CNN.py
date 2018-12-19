import preprocessing as preproc
import helpers
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Embedding, Activation, Flatten, GlobalMaxPooling1D
from keras.optimizers import Adam, SGD
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import text_representation as text_repr
from keras.models import load_model

start_time = time.time()

full_dataset = False
glove_pretrained = False

train_data, y, test_data = helpers.get_processed_data(full_dataset=full_dataset)
num_words = text_repr.define_num_words(train_data, full_dataset)
tweet_max_length = text_repr.define_tweet_max_len(train_data, full_dataset)

tokenizer = text_repr.fit_tokenizer(train_data, num_words)
train_data_seq = tokenizer.texts_to_sequences(train_data)
test_data_seq = tokenizer.texts_to_sequences(test_data)
train_data_seq = pad_sequences(train_data_seq, maxlen=tweet_max_length)
test_data_seq = pad_sequences(test_data_seq, maxlen=tweet_max_length)

model_cnn_02 = Sequential()
model_cnn_02.add(Embedding(num_words, 200, input_length=tweet_max_length))
model_cnn_02.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_02.add(GlobalMaxPooling1D())
model_cnn_02.add(Dense(256, activation='relu'))
model_cnn_02.add(Dropout(0.2))
model_cnn_02.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=5e-4, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
model_cnn_02.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(train_data_seq, y, test_size=0.15, random_state=42)

filepath_acc = "../data/intermediate/adam2.weights.best.hdf5"
checkpoint_acc = ModelCheckpoint(filepath_acc, monitor='val_acc', verbose=2, save_best_only=True, mode='max')

model_cnn_02.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=1, callbacks=[checkpoint_acc])


model = load_model("../data/intermediate/adam2.weights.best.hdf5")

helpers.predict_and_save(model, test_data_seq, 'CNN')

scores = []
for i in range(1,4):
    X_train, X_test, y_train, y_test = train_test_split(train_data_seq, y, test_size=10000, random_state=i)
    score = model.evaluate(X_test,y_test,verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    scores.append(score[1])
print('Average score:', sum(scores)/len(scores))
