import numpy as np
import re

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

def remove_unnecessary(text):
    return re.sub(r'<user>|<url>|\n', '', text)

def replace_emoji(text):
    rep_text = text
    rep_text = re.sub('>:\)|>:D|>:-D|>;\)|>:-\)|}:-\)|}:\)|3:-\)|3:\)', ' devil ', rep_text)
    rep_text = re.sub('O:-\)|0:-3|0:3|0:-\)|0:\)|0;^\)', ' angel ', rep_text)
    rep_text = re.sub(':\)+|\(+:|:-\)+|\(+-:|:\}| c : ', ' happy ', rep_text)
    # replace laugh icons with happy 
    rep_text = re.sub(':-D|:D|=D|=-D|8-D|8D|x-D|xD|X-D|=-D|:d|:-d|>:d|=3|=-3|:\'-\)|:\'\)|\(\':|\[:|:\]| : \) ', ' happy ', rep_text)    
    rep_text = re.sub('\)+:|:\(+|:-\(+|\)+-:|>:\[| : c |:\|', ' sad ', rep_text)
    rep_text = re.sub(':[ ]*\*|:-\*+|:x|: - \*', ' kiss ', rep_text, flags=re.I)
    rep_text = re.sub('<3', ' heart ', rep_text)
    rep_text = re.sub(';-\)|;\)|\*\)|\*-\)|;-\]|;]|;D|;\^\)', ' wink ', rep_text)
    rep_text = re.sub('>:P|:-P|:P|X-P|xp|=p|:b|:-b|;p| : p ', ' tongue ', rep_text, flags=re.I)
    rep_text = re.sub('>:O|:-O|:O| : O|: - o', ' surprise ', rep_text, flags=re.I)
    rep_text = re.sub(':-\||<:-\||>.<|:S|:\/|=\/|:\\\\| : - s ', ' skeptical ', rep_text)
    return rep_text