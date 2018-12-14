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
from glove import Glove, Corpus
import word2vec




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
    with open(DATA_INTERMEDIATE + file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in input_dict.items():
            writer.writerow([key, value])

def create_dict_from_csv(csv_path):
    '''create contractions dict from the corresponding csv'''
    reader = csv
    with open(csv_path) as f:
        reader = csv.reader(f)
        result = dict(reader)
        return result
    
def write_file(tweet_list, filename, is_test=False):
    with open(DATA_INTERMEDIATE + '{file}.txt'.format(file=filename), 'w') as f_out:
        if is_test:
            for index, tweet in enumerate(tweet_list):
                f_out.write('{},{}\n'.format(index, preproc.reduce_white_spaces(tweet)))
        else:
            for tweet in tweet_list:
                f_out.write('{}\n'.format(preproc.reduce_white_spaces(tweet)))
                
def get_processed_data(full_dataset=False, result_is_dataframe=False):
            
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
        df.loc[len(train_data):,'sentiment'] = -1
        return df
    y = np.zeros(shape=(len(train_data)))
    y[:len(pos_lines)] = 1
    y[len(pos_lines):] = -1
    
    return train_data, y, test_data

def transform_and_fit(train_data, y, test_data, text_representation='tfidf', 
                      ml_algorithm='LR', cross_val=False, full_dataset=False, predefined=False):
    
    if text_representation not in ['glove', 'word2vec']:
        clf = Pipeline([
            (text_representation, text_repr.get_transformer(text_representation)),
            (ml_algorithm, models.get_estimator(ml_algorithm))])
        print('Fit model...')
        clf.fit(train_data,y)
    else:
        
        if text_representation == 'glove':
            X_train, X_test = text_repr.train_glove_WE(train_data, test_data, full_dataset, predefined)
        elif text_representation == 'word2vec':
            X_train = word2vec.return_mean_vectorizer(train_data)
            X_test = word2vec.return_mean_vectorizer(test_data)
        else:
            raise ValueError('Invalid value text representation method')
                  
        print('Fit model...')       
        clf = models.get_estimator(ml_algorithm)
        clf.fit(X_train,y)
    
    if cross_val:
        print('Perform cross validation...')
        kf = KFold(n_splits=4, shuffle=True, random_state=1)
        if text_representation in ['glove', 'word2vec']:
            scores = cross_val_score(clf, X_train, y, cv=kf, n_jobs=2) 
        else:
            scores = cross_val_score(clf, train_data, y, cv=kf, n_jobs=-1)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        
    if text_representation in ['glove', 'word2vec']:
        return clf, X_test
    else:
        return clf, test_data

def predict_and_save(clf, X_test):

    y_pred = clf.predict(X_test)
    assert len(y_pred)==10000, 'Number of predictions: ' + str(len(y_pred))
    assert np.array_equal(np.unique(y_pred), [1, -1]) or np.array_equal(np.unique(y_pred), [-1, 1]) , (
            'Unique predicted labels: ' + str(np.unique(y_pred)))

    create_submission_csv(y_pred)
