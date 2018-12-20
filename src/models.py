#file that contains all the models that were used to reach the best one
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Embedding, Activation, Flatten, LSTM
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
import os
from keras.models import load_model

DATA_INTERMEDIATE = '../data/intermediate/'

def logistic_regression_count_vectorizer(lines, threshold_pos_neg):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(lines)

    y = np.zeros(shape=(len(lines)))
    y[:len(threshold_pos_negthreshold_pos_neg)] = 1
    y[len(threshold_pos_neg):] = -1

    clf = LogisticRegression().fit(X, y)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=kf)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

def return_model(model, num_words, embedding_matrix, tweet_max_length, input_shape=None):
    ''' provided the model that we want as parameter and its corresponding input parameters
        this function returns the corresponding model with the best possible parameters
    '''
    if model == 'LR':
        return LogisticRegression(max_iter=1000, solver='sag')
    elif model == 'RF':
        return RandomForestClassifier(n_estimators=100, max_depth=50)
    elif model == 'NB':
        return MultinomialNB()
    elif model == 'SVM':
        return SVC(probability=True)
    elif model == 'NN':
        return create_model_NN('NN', input_shape)
    elif model == 'LSTM':
        return create_best_LSTM_model(num_words, embedding_matrix, tweet_max_length)
    else:
        raise ValueError('Invalid value ML algorithm')
        
def create_model_NN(model_type='NN', input_shape=None):
    ''' create NN model '''
    if model_type == 'NN':
        model = Sequential()
        model.add(Dense(units=64, input_dim=input_shape, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == 'CNN':
        print('TODO CNN')

    elif model_type == 'LSTM':
        return create_best_LSTM_model()
    return model

def compile_model(model, optimizer='Adam'):
    ''' compile the NN model with the optimizer provided'''
    if optimizer == 'Adam':
        optimizer = Adam(lr=0.03, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    return model


def create_best_LSTM_model(num_words, embedding_matrix, tweet_max_length):
    ''' function that builds, compiles and returns our best LSTM model
    '''
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=200, weights=[embedding_matrix], 
                    input_length=tweet_max_length, trainable=True))
    model.add(LSTM(units=32, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(units=32, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(units=32, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(2, activation='softmax'))
    # optimizer = Adam(lr=5e-4, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def fit_NN_model(train_data_seq, y, filepath):
    best_model_file = DATA_INTERMEDIATE + filepath
    X_train, X_test, y_train, y_test = train_test_split(train_data_seq, y, test_size=0.2, random_state=42)
    checkpoint_acc = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    scores = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2, callbacks=[checkpoint_acc])

def best_model_is_provided(filepath):
    ''' returns True is the best model is provided'''
    if os.path.isfile(DATA_INTERMEDIATE + filepath):
        return True

def return_best_model(filepath):
    ''' returns the best model'''
    return load_model(DATA_INTERMEDIATE+filepath)
