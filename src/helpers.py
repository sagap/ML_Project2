import csv
import re
import preprocessing as preproc
import os
import numpy as np
import models
import text_representation as text_repr
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold
# from glove import Glove, Corpus
import word2vec
from keras.callbacks import ModelCheckpoint, EarlyStopping
import cross_validation as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA = '../data/'
DATA_TWITTER_DATASETS = '../data/twitter-datasets/'
DATA_INTERMEDIATE = '../data/intermediate/'

def create_submission_csv(y_pred):
    '''give predictions and create submission csv file'''
    with open(DATA + 'submission.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(range(1, len(y_pred) + 1), y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def write_dict(input_dict, file):
    ''' create csv file from dictionary'''
    with open(DATA_INTERMEDIATE + file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in input_dict.items():
            writer.writerow([key, value])

def create_dict_from_csv(csv_path):
    ''' create dict from the corresponding csv
        returns: the dictionary created
    '''
    reader = csv
    with open(csv_path) as f:
        reader = csv.reader(f)
        result = dict(reader)
        return result
    
def write_file(tweet_list, filename, is_test=False):
    '''  function that provided a list of tweets it writes them to a file and stores the file 
         under '/data/intermediate/'
         e.g. this function is invoked when the preprocessing is finished in order to save the processed tweets in files
    '''
    with open(DATA_INTERMEDIATE + '{file}.txt'.format(file=filename), 'w') as f_out:
        if is_test:
            for index, tweet in enumerate(tweet_list):
                f_out.write('{},{}\n'.format(index, preproc.reduce_white_spaces(tweet)))
        else:
            for tweet in tweet_list:
                f_out.write('{}\n'.format(preproc.reduce_white_spaces(tweet)))
                
def get_processed_data(full_dataset=False, result_is_dataframe=False):
    ''' 
        param: 
            full_dataset: full_dataset for True or small dataset for False  
            result_is_dataframe: if True returns the processed data in dataframe

        returns: processed data

        description: if processed files exist in 'data/intermediate/' they are fetched 
                     otherwise do_preprocessing() preprocesses raw tweets
    '''
    pos_prefix = 'train_pos'
    neg_prefix = 'train_neg'
    if full_dataset:
        pos_prefix = pos_prefix + '_full'
        neg_prefix = neg_prefix + '_full'
    test_prefix = 'test_data'
    
    train_pos_processed = DATA_INTERMEDIATE + pos_prefix + '_processed.txt'
    train_neg_processed = DATA_INTERMEDIATE + neg_prefix + '_processed.txt'
    test_processed = DATA_INTERMEDIATE + test_prefix + '_processed.txt'

    if not os.path.isfile(train_pos_processed):
        preproc.do_preprocessing(DATA_TWITTER_DATASETS + pos_prefix + '.txt')

    if not os.path.isfile(train_neg_processed):
        preproc.do_preprocessing(DATA_TWITTER_DATASETS + neg_prefix + '.txt')

    if not os.path.isfile(test_processed):
        preproc.do_preprocessing(DATA_TWITTER_DATASETS + test_prefix + '.txt', is_test=True)
    
    with open(train_pos_processed) as pos_in, open(train_neg_processed) as neg_in, open(test_processed) as test_in:
        pos_lines = pos_in.readlines()
        neg_lines = neg_in.readlines()
        test_lines = test_in.readlines()
        pos_in.close()
        neg_in.close()
        test_in.close()
    train_data = pos_lines + neg_lines
    #remove '\n' escape character
    train_data = [preproc.remove_new_line(tweet) for tweet in train_data]
    test_data = [preproc.remove_new_line(tweet).split(',', 1)[1] for tweet in test_lines]
    if result_is_dataframe:
        df = pd.DataFrame(train_data)
        df['sentiment'] = 1
        df.loc[len(train_data):,'sentiment'] = 0
        return df
    y = np.zeros(shape=(len(train_data)))
    y[:len(pos_lines)] = 1
    y[len(pos_lines):] = 0
    
    return train_data, y, test_data

def transform_and_fit(train_data, y, test_data, text_representation='tfidf', 
                      ml_algorithm='LR', cross_val=False, full_dataset=False, predefined=False):
    ''' '''
    if text_representation not in ['glove', 'word2vec']:
        if(ml_algorithm in ['NN', 'CNN']):
            X_train, X_test = text_repr.train_on_vectorizer(train_data, test_data, text_representation)
            if not cross_val :
                model = create_and_fit_NN_model(X_train, y, 3, ml_algorithm, True)
            # if cross_val == True then we perform cross-validation for the neural network
            else:
                model = create_and_fit_NN_model(X_train, y, 3, ml_algorithm, False)           
        else:
            model = Pipeline([
            (text_representation, text_repr.get_transformer(text_representation)),
            (ml_algorithm, models.get_estimator(ml_algorithm))])
            model.fit(train_data,y)
        print('Fit model...')
        
    else:
        if text_representation == 'glove':
            X_train, X_test = text_repr.train_glove_WE(train_data, test_data, full_dataset, predefined)
        elif text_representation == 'word2vec':
            X_train = word2vec.return_mean_vectorizer(train_data)
            X_test = word2vec.return_mean_vectorizer(test_data)
        else:
            raise ValueError('Invalid value text representation method')
                  
        print('Fit model...')     
        if ml_algorithm in ['NN', 'CNN']:
            if not cross_val :
                print('Fit the NN model')
                model = create_and_fit_NN_model(X_train, y, 3, ml_algorithm, True)
            else:
                model = create_and_fit_NN_model(X_train, y, 3, ml_algorithm, False)                
        else:
            model = models.get_estimator(ml_algorithm)
            model.fit(X_train,y)
    
    if cross_val:
        print('Perform cross validation...')
        if ml_algorithm in ['NN', 'CNN']:
            scores = cv.cross_validation_NN(model, X_train, y, 1)
            print_history(scores)
        else:
            if text_representation in ['glove', 'word2vec']:
                scores = cv.cross_validation(model, X_train, y , 4, 2)
            else:
                scores = cv.cross_validation(model, train_data, y , 4, 2)
            print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    if ml_algorithm in ['NN', 'CNN']:
        validate_model(X_train, y)
        return model, X_test
    if text_representation in ['glove', 'word2vec']:
        return model, X_test
    else:
        return model, test_data

def predict_and_save(model, X_test, ml_algorithm):
    ''' function that predicts 
        
    '''
    if ml_algorithm in ['NN', 'CNN', 'LSTM']:
        y_pred = model.predict_classes(X_test)
    else:
        y_pred = model.predict(X_test)
    y_pred[y_pred == 0] = -1
    assert len(y_pred)==10000, 'Number of predictions: ' + str(len(y_pred))
    assert np.array_equal(np.unique(y_pred), [1, -1]) or np.array_equal(np.unique(y_pred), [-1, 1]) , (
            'Unique predicted labels: ' + str(np.unique(y_pred)))

    create_submission_csv(y_pred)

def print_history(history):
    ''' function used to print the history after fit 
        to plot the accuracy and the loss of train-test set
    '''
    # plot accuracy of train-validation set, after fit to an Neural Network
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('plot_acc.png')
    # plot loss of train-validation set, after fit to an Neural Network
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig('plot_loss.png')

def checkpointing():
    ''' function that returns a list of callbacks for the fit function of keras model,
    '''
    filepath_acc = DATA_INTERMEDIATE+"best.weights.acc.hdf5"
    filepath_loss = DATA_INTERMEDIATE+"best.weights.loss.hdf5"
    checkpoint_acc = ModelCheckpoint(filepath_acc, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    checkpoint_loss = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max') 
    return [checkpoint_acc, checkpoint_loss, early_stop]

def batch_generator(X, y, batch_size):
    ''' function that is used for testing purposes
        generates batches from X, y according to the batch size
    '''
    batch_per_epoch = int(X.shape[0]/batch_size)
    index = np.arange(np.shape(y)[0])
    batches_x = [X[batch_size*i:batch_size*(i+1)] for i in range(batch_per_epoch)]
    batches_y = [y[batch_size*i:batch_size*(i+1)] for i in range(batch_per_epoch)]
    while(True):
        for counter in range(len(batches_x)):
            yield batches_x[counter], batches_y[counter]

def create_and_fit_NN_model(X, y, ml_algorithm, epochs, should_fit=False, callbacks=None):
    ''' creates and compiles the NN model and fits '''
    model = models.get_estimator(ml_algorithm, X.shape[1])
    model = models.compile_model(model, 'Adam')
    if should_fit:
        result = model.fit(X, y, epochs=epochs, batch_size=32, verbose=2, callbacks=callbacks)
    return model

def validate_model(X, y):
    ''' validate the model over 10k sample of the train set and save the best checkpoints'''
    print('Now validate model over 10k samples...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    #TODO load best model
    model_to_perform_validation = create_and_fit_NN_model(X_train, y_train, 'NN', 5, False)
    result = model_to_perform_validation.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=2, callbacks=checkpointing())
    
def load_best_model(X, y, ml_algorithm, filepath):
    model = models.get_estimator(ml_algorithm, X.shape[1])
    model.load_weights(DATA_INTERMEDIATE+"best_on_acc.hdf5")
    model = models.compile_model(model, 'Adam')
    return model
