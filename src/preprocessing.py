#preprocessing techniques
import numpy as np
import re
import helpers, string
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from math import log

#create slang dictionary
slang_dict = helpers.create_dict_from_csv('../twitter-datasets/slang.csv')
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
        pro_line = remove_tags(pro_line)
        pro_line = replace_slang(pro_line)
        pro_line = replace_contraction(pro_line)
        pro_line = replace_emoji(pro_line)
        pro_line = convert_to_lowercase(pro_line)
        pro_line = filter_digits(pro_line)
        pro_line = replace_elongated(pro_line)
        pro_line = separate_hashtags(pro_line)
        pro_line = remove_new_line(pro_line)
        processed_list.append(pro_line)

    filename = filepath.split('/')[-1][:-4] + '_processed'
    helpers.write_file(processed_list, filename)

def return_processed_trainset_and_y(full_dataset=True):
    if full_dataset:
        pos_file = '../twitter-datasets/train_pos_full_processed.txt'
        neg_file = '../twitter-datasets/train_neg_full_processed.txt'
    else:
        pos_file = '../twitter-datasets/train_pos_processed.txt'
        neg_file = '../twitter-datasets/train_neg_processed.txt'
    with open(pos_file) as pos_in, open(neg_file) as neg_in:
        pos_lines = pos_in.readlines()
        neg_lines = neg_in.readlines()
        pos_in.close()
        neg_in.close()
    lines = pos_lines + neg_lines
    #remove '\n' escape character
    lines = [remove_new_line(line) for line in lines]
    y = np.zeros(shape=(len(lines)))
    y[:len(pos_lines)] = 1
    y[len(pos_lines):] = -1
    return lines, y

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
    return word

def remove_tags(text):
    return re.sub(r'<user>|<url>', '', text)

def remove_new_line(text):
    return re.sub(r'\n', '', text)

def replace_emoji(text):
    rep_text = text
    rep_text = re.sub(':(-)?@', ' angry ', rep_text)
    rep_text = re.sub(':( )?\$', ' blushing ', rep_text)
    rep_text = re.sub('>:\)|>:D|>:-D|>;\)|>:-\)|}:-\)|}:\)|3:-\)|3:\)', ' devil ', rep_text) #done
    rep_text = re.sub('O:-\)|0:-3|0:3|0:-\)|0:\)|0;^\)', ' angel ', rep_text)
    rep_text = re.sub(':( )?\)+|\(+:|:-\)+|\(+-:|:\}| c : |:( )?O( )?\)|:( )?-( )?]|^( )?-( )?^', ' happy ', rep_text, flags=re.I)
    rep_text = re.sub(
        ':-D|:D|=D|=-D|8-D|8D|x-D|xD|X-D|=-D|:( )?d|:-d|>:d|=3|=-3|:\'-\)|:\'\)|\(\':|\[:|:\]', ' happy ', rep_text)    
    rep_text = re.sub('\)+:|:\(+|:-\(+|\)+-:|>:\[| : c |:\||:-\[', ' sad ', rep_text)
    rep_text = re.sub(':( )?\*|:( )?-( )?\*+|:x', ' kiss ', rep_text, flags=re.I) 
    rep_text = re.sub('<3', ' heart ', rep_text)
    rep_text = re.sub(';-\)|;\)|\*\)|\*-\)|;-\]|;]|;D|;\^\)', ' wink ', rep_text)
    rep_text = re.sub('>:P|:-P|:P|X-P|xp|=p|:b|:-b|;p| : p ', ' tongue ', rep_text, flags=re.I)
    rep_text = re.sub('>:O|:( )?-( )?O|:( )?O', ' surprise ', rep_text, flags=re.I) #done
    rep_text = re.sub(
    ':-\||<:-\||>( )?.( )?<|:( )?-( )?\/|:( )?-( )?\\\\|:S|:( )?\/|=\/|=( )?\\\\|:( )?\\\\|:( )?-( )?s|;( )?/|:( )?l ',
    ' skeptical ', rep_text) #done
    return rep_text

# def replace_numbers(text):
#     return re.sub('[-+:\*\\\\/x#]?[0-9]+', ' ', text)
def replace_numbers(text):
    # TODO decide if it will be empty or leave the <number>.
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

def replace_exclamation(tweet):
    return re.sub('!((\s)?!)+', ' !! ', tweet)

def pos_tag(tweet):
    tweet = nltk.word_tokenize(tweet.lower())
    return nltk.pos_tag(tweet)

def separate_hashtags(tweet):
    # TODO: Check if we want <hashtag> in the beginning
    # slightly worst than without it in tfidf-csv_local-small_dataset
    # but might be useful for NN
    new_tokens = []
    tokens = tweet.split(' ')
    for token in tokens:
        if token.startswith('#'): new_tokens.append(
            '<hashtag> ' + hashtag.split_hashtag_to_words(token))
        else: new_tokens.append(token)
    return ' '.join(new_tokens)

def convert_to_lowercase(text):
    '''
    Converts all words in the article into lower case
    
    Parameters:
    text: text string
    return: text string to lower case
    '''
    return text.lower()

def remove_singe_char(tweet):
    '''Does not remove char if it is in the beggining or the end of the tweet'''
    return re.sub('\s\w\s', ' ', tweet)

def replace_slang(tweet):
    return ' '.join([slang_dict[word] if word in slang_dict else word for word in tweet.split(' ')])

def replace_white_spaces(tweet):
    return re.sub('\s+', ' ', tweet)

def standardize_with_standrard_scaler(train_tweets):
    scaler = StandardScaler(with_mean=False)
    return scaler.fit_transform(train_tweets)

def filter_digits(tweet):
    """
    DESCRIPTION: 
                Filters digits from a tweet. Words that contain digits are not filtered.
    INPUT: 
                tweet: a tweet as a python string
    OUTPUT: 
                digit-filtered tweet as a python string
    """
    t = []
    for w in tweet.split():
        try:
            num = re.sub('[,\.:%_\-\+\*\/\%\_]', '', w)
            float(num)
            t.append("number")
        except:
            t.append(w)
    return (" ".join(t)).strip()

#Copyright (c) http://stackoverflow.com/users/1515832/generic-human (http://stackoverflow.com/a/11642687)

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open('../twitter-datasets/words-by-frequency.txt').read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)


def split_hashtag_to_words(hashtag):
    try: return infer_spaces(hashtag[1:]).strip()
    except: return hashtag[1:]

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)
    
    new_s = ''
    new_s = re.sub('[0-9]+', 'number', s)
    s = new_s
    
    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))

def separate_hashtags(tweet):
    # TODO: Check if we want <hashtag> in the beginning
    # slightly worst than without it in tfidf-csv_local-small_dataset
    # but might be useful for NN
    new_tokens = []
    tokens = tweet.split(' ')
    for token in tokens:
        if token.startswith('#'): new_tokens.append(
            '<hashtag> ' + split_hashtag_to_words(token))
        else: new_tokens.append(token)
    return ' '.join(new_tokens)
