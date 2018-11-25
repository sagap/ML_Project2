import numpy as np

def load_embeddings():
    '''load embeddings provided''' 
    np.load('../twitter-datasets/embeddings.npy')

def create_dict_from_provided_vocabulary():
    '''create dict for test purposes'''
    vocab_dict = {}
    embeddings = np.load('../twitter-datasets/embeddings.npy')
    with open('../twitter-datasets/vocab_cut.txt', 'r') as vocab_in:
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