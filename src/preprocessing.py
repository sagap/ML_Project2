#preprocessing techniques
import re
import helpers, string
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

#create contractions dictionary
contractions = helpers.create_dict_from_csv('utils/contractions.csv')
#compile regular expressions from dictionary keys
contractions_regexp = re.compile('(%s)' % '|'.join(contractions.keys()))
# compile regex to find 
elongated_regex = re.compile(r"(.)\1{1}")

def do_preprocessing(filepath, test_file=False):
    with open(filepath, 'r') as f_in:
        lines = f_in.readlines()
        f_in.close()    
    
    if test_file:
        lines = [line.split(',', 1)[1] for line in lines]
    
    processed_list = []
    for line in tqdm(lines):
        pro_line = line
        pro_line = remove_unnecessary(pro_line)
        pro_line = replace_contraction(pro_line)
        pro_line = replace_numbers(pro_line)
        pro_line = replace_emoji(pro_line)
        pro_line = replace_elongated(pro_line)
#        pro_line = stemming_using_Porter(pro_line)
        pro_line = remove_stopwords(pro_line)
#        pro_line = lemmatize_verbs(pro_line)
        processed_list.append(pro_line)

    filename = filepath.split('/')[-1][:-4] + '_processed'
    helpers.write_file(processed_list, filename)

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
    
def replace_elongated(line):
    ''' function that takes a line as input and checks for elongated words'''
    fixed_tweet = fixed_tweet = [replace_elongated_word(word) if len(wordnet.synsets(word))==0 else word for word in line.split()]
    return ' '.join(fixed_tweet)    

def replace_elongated_word(word):
    ''' function that provided an elongated word, returns its proper form
        based on wordnet of nltk to find the proper word without elongation
	e.g. Goooodd -> Good
    '''
    if wordnet.synsets(word):
        return word + word
    elif is_elongated(word):
        return replace_elongated_word(elongated_regex.sub(r'\1', word))
    return word + word

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

# def replace_numbers(text):
#     return re.sub('[-+:\*\\\\/x#]?[0-9]+', ' ', text)
def replace_numbers(text):
    return re.sub('[0-9]', ' ', text)

def stemming_using_Porter(text):
    '''function that uses PorterStemmer on a list of tweets'''
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split(' ')])

def remove_stopwords(text):
    ''' function that removes stop words based on the corpus of nltk '''
    return ' '.join([word for word in text.split(' ') if word not in stopwords.words('english')])

def lemmatize_verbs(text):
    ''' function that lemmatizes words based on WordNetLemmatizer'''
    lemmatizer = WordNetLemmatizer()
    lemmatized_tweet = []
    [lemmatized_tweet.append(lemmatizer.lemmatize(word, pos='v')) for word in text.split(' ')]
    return ' '.join(lemmatized_tweet)

def replace_repeated_punctuation(tweet):
    tokenizer = RegexpTokenizer(r'\w+')
    return " ".join(tokenizer.tokenize(tweet))

def remove_punctuation(text):
    ''' function that is executed after 'replace_emoji' to remove all the uneccesary punctuations'''
    return ' '.join([element for element in text.split(' ') if element not in string.punctuation])

def replace_exclamation(text):
    return re.sub('![ !]+', ' !!! ', text)

def pos_tag(tweet):
    tweet = nltk.word_tokenize(tweet.lower())
    return nltk.pos_tag(tweet)

def separate_hashtags():
    # TODO
    return
def convert_to_lowercase(text):
    '''
    Converts all words in the article into lower case
    
    Parameters:
    text: text string
    return: text string to lower case
    '''
    return text.lower()

