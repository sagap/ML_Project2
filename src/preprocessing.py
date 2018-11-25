#preprocessing techniques
import re
import helpers
from nltk.corpus import wordnet

#create contractions dictionary
contractions = helpers.create_dict_from_csv('utils/contractions.csv')
#compile regular expressions from dictionary keys
contractions_regexp = re.compile('(%s)' % '|'.join(contractions.keys()))
# compile regex to find 
elongated_regex = re.compile(r"(.)\1{1}")

def replace_contraction(tweet):
    '''function that replaces possible contractions in tweets
        based on the contractions dictionary
    '''
    def replace(match):
        return contractions[match.group(0)]
    return contractions_regexp.sub(replace, tweet)

def is_elongated(word):
    ''' function that checks if a word is elongated'''
    return elongated_regex.search(word) != None
    
def replace_elongated_word(word):
    ''' function that provided an elongated word, returns its proper form
        based on wordnet of nltk to find the proper word without elongation
	e.g. Goooodd -> Good
    '''
    if wordnet.synsets(word):
        return word
    elif is_elongated(word):
        return replace_elongated_word(elongated_regex.sub(r'\1', word))
    return word

# for debugging reasons
def countElongated(text):
    """ Input: a text, Output: how many words are elongated """
    return len([word for word in text.split() if elongated_regex.search(word)])