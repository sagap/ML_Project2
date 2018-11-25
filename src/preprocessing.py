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

def remove_unnecessary(text):
    return re.sub(r'<user>|<url>|\n', '', text)

def replace_emoji(text):
    rep_text = ' ' + text + ' '
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

def replace_numbers(text):
    return re.sub('[-+:\*\\\\/x#]?[0-9]+', ' number ', text)

def write_file(_list, filename):
    with open('../twitter-datasets/{file}.txt'.format(file=filename), 'w') as f_out:
        for line in _list:
            f_out.write("%s\n" % line.strip())