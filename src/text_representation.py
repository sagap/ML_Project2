import preprocessing as preproc
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# from glove import Glove, Corpus

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

DATA_UTILS = '../data/utils/'
DATA_INTERMEDIATE = '../data/intermediate/'
DATA_TWITTER_DATASETS = '../data/twitter-datasets/'

def loadGloveModel(gloveFile):
    ''' function that loads word embeddings derived from glove file
    '''
    print("Loading Glove Word Embeddings")
    with open(gloveFile,'r') as f:
        lines = f.readlines()       
        word_embeddings = {}
        for line in lines:
            word = line.split(' ', 1)[0]
            embedding_vector = np.array([float(x) for x in line.split(' ', 1)[1].split()])
            word_embeddings[word] = embedding_vector
        return word_embeddings

def create_pad_sequences_train_test(tokenizer, train_data, test_data, tweet_max_length):
        ''' params: tokenizer: Keras Tokinizer
            train_data: processed train tweets
            test_data: processed test tweets
            tweet_max_length: maximum length of each tweet

            returns: the sequences of train and test processed tweets
        '''
        train_data_seq = create_sequences(train_data, tokenizer)
        test_data_seq = create_sequences(test_data, tokenizer)
        train_data_seq = pad_sequences(train_data_seq, maxlen=tweet_max_length)
        test_data_seq = pad_sequences(test_data_seq, maxlen=tweet_max_length)
        return train_data_seq, test_data_seq

def fit_tokenizer(train_data, num_words):
    ''' params: train_data: train set tweets
                num_words: number of words in vocabulary
        returns: keras Tokenizer

        description: Keras Tokenizer splits tweets in words
    '''
    tokenizer = Tokenizer(num_words=num_words, filters='', lower=False, split=' ',
                      char_level=False, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(train_data)
    return tokenizer

def get_embedding_matrix(tokenizer, word_embeddings, num_words):
    ''' function that returns the embedding matrix  
    '''
    embedding_matrix = np.zeros((num_words, 200))
    for word, i in tokenizer.word_index.items():
        if i >= num_words:
            continue
        if word in word_embeddings:
            embedding_vector = word_embeddings[word]
        else:
            embedding_vector = np.zeros(200)
        embedding_matrix[i] = embedding_vector
        
    return embedding_matrix

def define_tweet_max_len(train_data, full_dataset):
    ''' function that is used to find the maximum length of a tweet,
        of course we choose an optimal value, depending on the size of the dataset given as parameter
    '''
    if full_dataset:
        tweet_max_length = 130
    else:
        tweet_max_length = 80
        
    length = []
    for x in train_data:
        length.append(len(x.split()))    
    print('{0}: max sentence length found, {1}: will be used'.format(max(length), tweet_max_length))
    
    return tweet_max_length
    

def define_num_words(train_data, full_dataset):
    ''' function that uses CounVectorizer to find the number of words of the vocabulary
        then then return the number of words that we choose to use, according to the size of the dataset
    '''
    if full_dataset:
        num_words = 120000
    else:
        num_words = 30000
    vectorizer = CountVectorizer(min_df=2)
    count_vect = vectorizer.fit_transform(train_data)
    vocabulary = vectorizer.get_feature_names()
    no_vocabulary = len(vocabulary)
    print('{0}: num of words , {1}: will be used'.format(no_vocabulary, num_words))
    
    return num_words

def create_sequences(data, tokenizer):
    ''' provided the tokenizer, the data
        this function returns the corrresponding sequences
    '''
    sequences = tokenizer.texts_to_sequences(data)
    return sequences

def train_on_vectorizer(train_data, test_data, transformer_uri):
    vectorizer = get_transformer(transformer_uri)
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_test

def get_transformer(transformer_uri):
    if transformer_uri == 'tfidf':
        return TfidfVectorizer(stop_words=None, ngram_range=(1,2), sublinear_tf=True, max_features=200)
    elif transformer_uri == 'count_vect':
        return CountVectorizer()
    else:
        raise ValueError('Invalid value for transformer')

def load_embeddings():
    '''load embeddings provided''' 
    np.load(DATA_INTERMEDIATE + 'embeddings.npy')

def create_dict_from_provided_vocabulary():
    '''create dict for test purposes'''
    vocab_dict = {}
    embeddings = np.load(DATA_INTERMEDIATE + 'embeddings.npy')
    with open(DATA_INTERMEDIATE + 'vocab_cut.txt', 'r') as vocab_in:
        vocab_lines = vocab_in.readlines()
        line_counter = 0
        for line in vocab_lines:
            vocab_dict[line.replace('\n', '')] = embeddings[line_counter]
            line_counter += 1
    return vocab_dict

def create_tweet_features(path_to_file, vocab_dict, shape_of_word_embeddings):
    '''give path to file and vocabulary dictionary
        and return features of the tweets'''
    features = np.empty(shape=shape_of_word_embeddings)
    with open(path_to_file, 'r') as file_in:
        lines = file_in.readlines()    
        for line in lines:
            tweet = line.replace('\n', '').split(' ')
            tweet_features = np.zeros(shape=(len(tweet), shape_of_word_embeddings))
            for i in range(0, len(tweet)):
                if tweet[i] in vocab_dict:
                    tweet_features[i] = vocab_dict[tweet[i]]
            if np.count_nonzero(tweet_features) == 0:
                continue
            tweet_features = np.mean(tweet_features, axis=0)
            features = np.vstack((features, tweet_features))
    return features

def create_test_tweet_features(path_to_file, vocab_dict, shape_of_word_embeddings):
    '''give path to file and vocabulary dictionary
        and return features of the tweets'''
    features = np.empty(shape=shape_of_word_embeddings)
    with open(path_to_file, 'r') as file_in:
        lines = file_in.readlines()    
        for line in lines:
            tweet = line.replace('\n', '').split(' ')
            tweet_features = np.zeros(shape=(len(tweet), shape_of_word_embeddings))
            for i in range(0, len(tweet)):
                if tweet[i] in vocab_dict:
                    tweet_features[i] = vocab_dict[tweet[i]]
            tweet_features = np.mean(tweet_features, axis=0)
            features = np.vstack((features, tweet_features))
    return features

def create_feature_representation(feature_representation, max_features=1000):
    ''' the parameter feature_representation specifices which Vectorizer,
        will be used to extract the features from tweets
    '''
    if feature_representation == 'CountVectorizer':
        return CountVectorizer()
    elif feature_representation == 'TfidfVectorizer':
        return TfidfVectorizer(stop_words=None, ngram_range=(1,2), sublinear_tf=True, max_features=max_features)
