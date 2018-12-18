import preprocessing as preproc
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from glove import Glove, Corpus

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

DATA_UTILS = '../data/utils/'
DATA_INTERMEDIATE = '../data/intermediate/'
DATA_TWITTER_DATASETS = '../data/twitter-datasets/'


def define_tweet_max_len(train_data, full_dataset):
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

def get_features(vectorizer_uri, train_data, test_data, full_dataset, num_features=None):
    if vectorizer_uri == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1,2), sublinear_tf=True, max_features=num_features)
        X_train = vectorizer.fit_transform(train_data)
        X_test = vectorizer.transform(test_data)
    elif vectorizer_uri == 'count':
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(train_data)
        X_test = vectorizer.transform(test_data)
    elif vectorizer_uri == 'glove':
        word_embeddings = train_glove_WE(
            train_data, full_dataset, predefined=True, build_embeddings=True)
        X_train = get_glove_features(train_data, word_embeddings)
        X_test = get_glove_features(test_data, word_embeddings)
    else:
        raise ValueError('Invalid value for transformer')
    return X_train, X_test
        
# def TE_from_concat_WE(data, vocab_dict, tweet_max_length, num_of_features):
    
#     tokenizer = Tokenizer(num_words=tweet_max_length)
#     tokenizer.fit_on_texts(data)
#     sequences = tokenizer.texts_to_sequences(data)
    
#     embedding_matrix = np.zeros((tweet_max_length, num_of_features))
#     for word, i in tokenizer.word_index.items():
#         if i >= tweet_max_length:
#             continue
#         embedding_vector = vocab_dict[word]
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector

# def TE_from_mean_WE(tweet_list, vocab_dict, num_of_features):

#     all_features = np.empty(shape=(len(tweet_list),num_of_features))
    
#     for index, tweet in enumerate(tweet_list):
#         words_found = 0
#         tweet_features = np.zeros(num_of_features)
#         for i in range(0, len(tweet)):
#             if tweet[i] in vocab_dict:
#                 # TODO +=
#                 tweet_features += vocab_dict[tweet[i]]
#                 words_found += 1
#         if words_found > 0:
#             tweet_features = tweet_features / words_found
#         all_features[index] = tweet_features
        
#     return all_features


# def train_glove_WE(train_data, test_data, 
#         full_dataset, tweet_max_length=1, num_features=200, predefined=True, build_embeddings=True):
    
#     assert predefined or build_embeddings, (
#         'At least one between "predefined, build_embeddings" should be assinged True')
    
#     glove_components = num_features=200
#     learning_rate=0.05
#     epochs=50
#     no_threads=4

#     if full_dataset:
#         file_postfix = '_big'
#     else:
#         file_postfix = '_small'

#     print('Estimate the word embeddings...')

#     glove_file = DATA_INTERMEDIATE + 'glove' + file_postfix + '.pickle'
#     glove_predefined_file = DATA_INTERMEDIATE + 'glove_predefined.pickle'
#     glove_pretrained_stanford = DATA_TWITTER_DATASETS + 'glove.twitter.27B.200d.txt'

#     word_embeddings_predifined = dict()
#     word_embeddings_corpus = dict()

#     if predefined:
#         if os.path.isfile(glove_predefined_file):
#             print('Load saved instance with predefined tweet embeddings...')
#             glove_predefined = Glove()
#             glove_predefined = glove_predefined.load(glove_predefined_file)
#         else:      
#             print('Create glove instance from pretrained twitter embeddings...')
#             glove_predefined = Glove()
#             glove_predefined = glove_predefined.load_stanford(glove_pretrained_stanford)
#             glove_predefined.save(glove_predefined_file)
            
#         for word, index in glove_predefined.dictionary.items():
#             word_embeddings_predifined[word] = np.array(glove_predefined.word_vectors[index])

    
#     if build_embeddings:           
#         corpus_file = DATA_INTERMEDIATE + 'corpus' + file_postfix + '.pickle'
#         if os.path.isfile(corpus_file):
#             print('Load the cooccurrence matrix...')
#             corpus = Corpus()
#             corpus = corpus.load(corpus_file)
#         else:
#             print('Construct the cooccurrence matrix...')
#             corpus = Corpus()
#             tokenized_data = [preproc.replace_white_spaces(tweet).split(' ') for tweet in train_data]
#             corpus.fit(tokenized_data)
#             corpus.save(corpus_file)

#         if os.path.isfile(glove_file):
#             print('Load glove instance built on our training set...')
#             glove = Glove()
#             glove = glove.load(glove_file)
#         else:
#             print('Build word embeddings on our training set...')        
#             glove = Glove(no_components=glove_components, learning_rate=learning_rate)
#             glove.fit(corpus.matrix, epochs=epochs, no_threads=no_threads, verbose=True)
#             glove.add_dictionary(corpus.dictionary)
#             glove.save(glove_file)
        
#         for word, index in glove.dictionary.items():
#             word_embeddings_corpus[word] = np.array(glove.word_vectors[index])

#     # merging word embeddings
#     # if in both sets keep the predifined from stanford
#     for key in word_embeddings_corpus:
#         if key not in word_embeddings_predifined:
#             word_embeddings_predifined[key] = word_embeddings_corpus[key]
            
#     if word_embeddings_predifined:
#         word_embeddings = {key: word_embeddings_predifined[key] for key in word_embeddings_predifined}
#     else:
#         word_embeddings = {key: word_embeddings_corpus[key] for key in word_embeddings_corpus}

        
#     if glove_mean:    
#         print('Build tweet embeddings...')
#         X_train = TE_from_mean_WE(train_data, word_embeddings, glove_components)
#         X_test = TE_from_mean_WE(test_data, word_embeddings, glove_components)            
#     else:
#         X_train = TE_from_concat_WE(train_data, word_embeddings, tweet_max_length, glove_components)
#         X_test = TE_from_concat_WE(test_data, word_embeddings, tweet_max_length, glove_components)

    
#     return X_train, X_test

def train_glove_WE(train_data, full_dataset, 
     num_features=200, predefined=True, build_embeddings=True):
    
    assert predefined or build_embeddings, (
        'At least one between "predefined, build_embeddings" should be assinged True')
    
    glove_components = num_features=200
    learning_rate=0.05
    epochs=50
    no_threads=4

    if full_dataset:
        file_postfix = '_big'
    else:
        file_postfix = '_small'

    print('Estimate the word embeddings...')

    glove_file = DATA_INTERMEDIATE + 'glove' + file_postfix + '.pickle'
    glove_predefined_file = DATA_INTERMEDIATE + 'glove_predefined.pickle'
    glove_pretrained_stanford = DATA_TWITTER_DATASETS + 'glove.twitter.27B.200d.txt'

    word_embeddings_predifined = dict()
    word_embeddings_corpus = dict()

    if predefined:
        if os.path.isfile(glove_predefined_file):
            print('Load saved instance with predefined tweet embeddings...')
            glove_predefined = Glove()
            glove_predefined = glove_predefined.load(glove_predefined_file)
        else:      
            print('Create glove instance from pretrained twitter embeddings...')
            glove_predefined = Glove()
            glove_predefined = glove_predefined.load_stanford(glove_pretrained_stanford)
            glove_predefined.save(glove_predefined_file)
            
        for word, index in glove_predefined.dictionary.items():
            word_embeddings_predifined[word] = np.array(glove_predefined.word_vectors[index])

    
    if build_embeddings:           
        corpus_file = DATA_INTERMEDIATE + 'corpus' + file_postfix + '.pickle'
        if os.path.isfile(corpus_file):
            print('Load the cooccurrence matrix...')
            corpus = Corpus()
            corpus = corpus.load(corpus_file)
        else:
            print('Construct the cooccurrence matrix...')
            corpus = Corpus()
            tokenized_data = [preproc.replace_white_spaces(tweet).split(' ') for tweet in train_data]
            corpus.fit(tokenized_data)
            corpus.save(corpus_file)

        if os.path.isfile(glove_file):
            print('Load glove instance built on our training set...')
            glove = Glove()
            glove = glove.load(glove_file)
        else:
            print('Build word embeddings on our training set...')        
            glove = Glove(no_components=glove_components, learning_rate=learning_rate)
            glove.fit(corpus.matrix, epochs=epochs, no_threads=no_threads, verbose=True)
            glove.add_dictionary(corpus.dictionary)
            glove.save(glove_file)
        
        for word, index in glove.dictionary.items():
            word_embeddings_corpus[word] = np.array(glove.word_vectors[index])

    # merging word embeddings
    # if in both sets keep the predifined from stanford
    for key in word_embeddings_corpus:
        if key not in word_embeddings_predifined:
            word_embeddings_predifined[key] = word_embeddings_corpus[key]
            
    if word_embeddings_predifined:
        word_embeddings = {key: word_embeddings_predifined[key] for key in word_embeddings_predifined}
    else:
        word_embeddings = {key: word_embeddings_corpus[key] for key in word_embeddings_corpus}
        
    return word_embeddings

def get_glove_features(data, word_embeddings, glove_components=200):
    print('Build tweet embeddings...')
    X_matrix = TE_from_mean_WE(data, word_embeddings, glove_components)
    
    return X_matrix

def TE_from_mean_WE(tweet_list, vocab_dict, num_of_features):

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

def get_embedding_matrix(train_data, word_embeddings, num_words):
    tokenizer = Tokenizer(num_words=num_words, filters='', lower=False, split=' ',
                          char_level=False, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(train_data)
    sequences = tokenizer.texts_to_sequences(train_data)

    embedding_matrix = np.zeros((num_words, 200))
    for word, i in tokenizer.word_index.items():
        if i >= num_words:
            continue
        embedding_vector = word_embeddings[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    return embedding_matrix, sequences

def create_sequences(data, num_words):
    tokenizer = Tokenizer(num_words=num_words, filters='', lower=False, split=' ',
                      char_level=False, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(train_data)
    sequences = tokenizer.texts_to_sequences(train_data)
    
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
