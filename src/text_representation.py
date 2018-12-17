import preprocessing as preproc
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from glove import Glove, Corpus

DATA_UTILS = '../data/utils/'
DATA_INTERMEDIATE = '../data/intermediate/'

def TE_from_WE(tweet_list, vocab_dict, num_of_features):

    all_features = np.empty(shape=(len(tweet_list),num_of_features))
    
    for index, tweet in enumerate(tweet_list):
        words_found = 0
        tweet_features = np.zeros(num_of_features)
        for i in range(0, len(tweet)):
            if tweet[i] in vocab_dict:
                # TODO +=
                tweet_features += vocab_dict[tweet[i]]
                words_found += 1
        if words_found > 0:
            tweet_features = tweet_features / words_found
        all_features[index] = tweet_features
        
    return all_features

def train_glove_WE(train_data, test_data, full_dataset, predefined=False):
     
    glove_components = 300
    learning_rate=0.05
    epochs=5
    no_threads=4

    if full_dataset:
        file_postfix = '_big'
    else:
        file_postfix = '_small'

    print('Construct the cooccurrence matrix...')
    corpus_file = DATA_INTERMEDIATE + 'corpus' + file_postfix + '.pickle'
    if os.path.isfile(corpus_file):
        corpus = Corpus()
        corpus = corpus.load(corpus_file)
    else:
        corpus = Corpus()
        tokenized_data = [preproc.replace_white_spaces(tweet).split(' ') for tweet in train_data]
        corpus.fit(tokenized_data, window = 5)
    # TODO: add this
    # corpus.save(corpus_file)

    print('Estimate the word embeddings...')
    glove_file = DATA_INTERMEDIATE + 'glove' + file_postfix + '.pickle'
    # if os.path.isfile(glove_file):
    #         glove = Glove()
    #     glove = glove.load(glove_file)
    # else:
    if predefined:
        print('predefined')
        glove = Glove()
        glove = glove.load_stanford('../data/twitter-datasets/glove.twitter.27B/glove.twitter.27B.200d.txt')
    else:        
        glove = Glove(no_components=glove_components, learning_rate=learning_rate)
        glove.fit(corpus.matrix, epochs=epochs, no_threads=no_threads, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        
    glove.save(glove_file)

    print('Build word embeddings...')
    word_embeddings = dict()
    for word, index in glove.dictionary.items():
        word_embeddings[word] = np.array(glove.word_vectors[index])

    print('Build tweet embeddings...')
    X_train = TE_from_WE(train_data, word_embeddings, glove_components)
    X_test = TE_from_WE(test_data, word_embeddings, glove_components)
    
    return X_train, X_test

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
	# elif feature_representation == 'Word2Vec'

	# TODO
