#preprocessing techniques
import numpy as np
import re
import string
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from math import log
import pandas as pd
import nltk
# nltk.download('sentiwordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag

import helpers

DATA_UTILS = '../data/utils/'
DATA_INTERMEDIATE = '../data/intermediate/'

#create slang dictionary
slang_dict = helpers.create_dict_from_csv(DATA_UTILS + 'slang.csv')
#create contractions dictionary
contractions = helpers.create_dict_from_csv(DATA_UTILS + 'contractions.csv')
#compile regular expressions from dictionary keys
contractions_regexp = re.compile('(%s)' % '|'.join(contractions.keys()))
# compile regex to find 
elongated_regex = re.compile(r"(.)\1{1}")

def do_preprocessing(filepath, is_test=False):
    ''' 
        params: filepath: which file of tweets to perform preprocessing on
        
    '''    
    initial_file = filepath.split('/')[-1]
    new_file = initial_file[:-4] + '_processed'
    print('preprocessing', initial_file)
    
    with open(filepath, 'r') as f_in:
        lines = f_in.readlines()
        f_in.close()    
    
    if is_test:
        lines = [line.split(',', 1)[1] for line in lines]
    
    processed_list = []
    for line in lines:
        pro_line = line
        pro_line = remove_new_line(pro_line)
        pro_line = replace_slang(pro_line)
        pro_line = replace_contraction(pro_line)        
        pro_line = remove_tags(pro_line)
        pro_line = replace_emoji(pro_line)
        pro_line = convert_to_lowercase(pro_line)
        # pro_line = lemmatize_verbs(pro_line)
        # pro_line = stemming_using_Porter(pro_line)
        # pro_line = remove_stopwords(pro_line)
        pro_line = separate_hashtags(pro_line)
        pro_line = replace_numbers(pro_line)
        pro_line = replace_elongated(pro_line)
        pro_line = add_sentiment(pro_line)
        processed_list.append(pro_line)
    
    helpers.write_file(processed_list, new_file, is_test)

def return_processed_trainset_and_y(full_dataset=True, result_is_dataframe=False):
    if full_dataset:
        pos_file = DATA_INTERMEDIATE + 'train_pos_full_processed.txt'
        neg_file = DATA_INTERMEDIATE + 'train_neg_full_processed.txt'
    else:
        pos_file = DATA_INTERMEDIATE + 'train_pos_processed.txt'
        neg_file = DATA_INTERMEDIATE + 'train_neg_processed.txt'
    with open(pos_file) as pos_in, open(neg_file) as neg_in:
        pos_lines = pos_in.readlines()
        neg_lines = neg_in.readlines()
        pos_in.close()
        neg_in.close()
    lines = pos_lines + neg_lines
    #remove '\n' escape character
    lines = [remove_new_line(line) for line in lines]
    if result_is_dataframe:
        df = pd.DataFrame(lines)
        df['sentiment'] = 1
        df.loc[len(pos_lines):,'sentiment'] = 0
        return df
    y = np.zeros(shape=(len(lines)))
    y[:len(pos_lines)] = 1
    y[len(pos_lines):] = 0
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
    if len(re.findall('[^a-z]', word))!=0:
        return word    
    if wordnet.synsets(word):
        return word + " " + word 
    elif is_elongated(word):
        return replace_elongated_word(elongated_regex.sub(r'\1', word))
    return word

def remove_tags(text):
    return re.sub(r'<user>|<url>', '', text)

def remove_new_line(text):
    return re.sub(r'\n', '', text)

# def replace_emoji(text):
#     rep_text = text
#     rep_text = re.sub(':(-)?@', ' angryface emoji ', rep_text)
#     rep_text = re.sub(':( )?\$', ' blushing face emoji ', rep_text)
#     rep_text = re.sub('>:\)|>:D|>:-D|>;\)|>:-\)|}:-\)|}:\)|3:-\)|3:\)', ' devil emoji ', rep_text) #done
#     rep_text = re.sub('O:-\)|0:-3|0:3|0:-\)|0:\)|0;^\)', ' angelface emoji ', rep_text)
#     rep_text = re.sub(':( )?‑( )?\||:( )?\|', 'straightface emoji ', rep_text)
#     rep_text = re.sub(':( )?\)+|\(+:|:-\)+|\(+-:|:\}| c : |:( )?O( )?\)|:( )?-( )?]|^( )?-( )?^', ' happyface emoji ',
#                       rep_text, flags=re.I)
#     rep_text = re.sub(
#         ':-D|:D|=D|=-D|8-D|8D|x-D|xD|X-D|=-D|:( )?d|:-d|>:d|=3|=-3|:\'-\)|:\'\)|\(\':|\[:|:\]', ' happyface emoji ', rep_text)    
#     rep_text = re.sub('\)+:|:\(+|:-\(+|\)+-:|>:\[| : c |:\||:-\[', ' sadface emoji ', rep_text)
#     rep_text = re.sub(':( )?\*|:( )?-( )?\*+|:x', ' kissyface emoji ', rep_text, flags=re.I) 
#     rep_text = re.sub('<3', ' heart emoji ', rep_text)
#     rep_text = re.sub(';( )?-( )?\)|;( )?\)|\*\)|\*-\)|;( )?-( )?\]|;( )?]|;( )?D|;\^\)', ' winkyface emoji ', rep_text)
#     rep_text = re.sub('>:P|:-P|:P|X-P|xp|=p|:b|:-b|;p| : p ', ' tongue emoji ', rep_text, flags=re.I)
#     rep_text = re.sub('>:O|:( )?-( )?O|:( )?O', ' surprise emoji ', rep_text, flags=re.I) #done
#     rep_text = re.sub(
#     '<:-\||>( )?.( )?<|:( )?-( )?\/|:( )?-( )?\\\\|:S|:( )?\/|=\/|=( )?\\\\|:( )?\\\\|:( )?-( )?s|;( )?/|:( )?l ',
#     ' skeptical emoji ', rep_text) #done
#     return rep_text

def replace_emoji(text):
    rep_text = text
    rep_text = re.sub(':( )?@|:( )?-( )?@', ' angryface emoji ', rep_text)
    rep_text = re.sub(':( )?\$', ' blushing face emoji ', rep_text)
    rep_text = re.sub('>:\)|>:D|>:-D|>;\)|>:-\)|}:-\)|}:\)|3( )?:( )?-( )?\)|3( )?:( )?\)', ' devil emoji ',
                      rep_text) #done
    rep_text = re.sub('O( )?:( )?-( )?\)|0( )?:( )?-( )?3|0( )?:( )?3|0( )?:( )?-( )?\)|0( )?:( )?\)|0( )?;( )?^\)',
                      ' angelface emoji ', rep_text)
    rep_text = re.sub(':( )?-( )?\||:( )?\|', 'straightface emoji ', rep_text)
    rep_text = re.sub(':( )?\)+|\(+( )?:|:( )?-( )?\)+|\(+( )?-( )?:|:( )?\}| c : |:( )?O( )?\)|:( )?-( )?]|^( )?-( )?^',
                      ' happyface emoji ', rep_text, flags=re.I)
    rep_text = re.sub(
        ':( )?-( )?D|:( )?D|=( )?D|=( )?-( )?D|8( )?-( )?D|8( )?D|x( )?-( )?D|x( )?D|X( )?-( )?D|=( )?-( )?D',
        ' happyface emoji ', rep_text)
    rep_text = re.sub(':( )?d|:( )?-d|>:d|=( )?3|=( )?-( )?3|:\'-\)|:\'\)|\(\':|\[:|:( )?\]',
                      ' happyface emoji ', rep_text)    
    rep_text = re.sub('\)+:|:\(+|:( )?-( )?\(+|\)+-:|>:\[| : c |:( )?-( )?\[', ' sadface emoji ', rep_text)
    rep_text = re.sub(':( )?\*|:( )?-( )?\*+|:x', ' kissyface emoji ', rep_text, flags=re.I) 
    rep_text = re.sub('<( )?3', ' heart emoji ', rep_text)
    rep_text = re.sub(';( )?-( )?\)|;( )?\)|\*\)|\*-\)|;( )?-( )?\]|;( )?]|;( )?D|;\^\)', ' winkyface emoji ', rep_text)
    rep_text = re.sub('>( )?:( )?P|:( )?-( )?P|:( )?P|X( )?-( )?P|x( )?p|=( )?p|:( )?b|:-b|;( )?p| : p ',
                      ' tongue emoji ', rep_text, flags=re.I)
    rep_text = re.sub('>:O|:( )?-( )?O|:( )?O', ' surprise emoji ', rep_text, flags=re.I) #done
    rep_text = re.sub(
    '<:-\||>( )?.( )?<|:( )?-( )?\/|:( )?-( )?\\\\|:S|:( )?\/|=\/|=( )?\\\\|:( )?\\\\|:( )?-( )?s|;( )?/|:( )?l ',
    ' skeptical emoji ', rep_text) #done
    return rep_text

def represents_float(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

# def replace_numbers(tweet):
#     new_tokens = []
#     tokens = tweet.split(' ')
#     for token in tokens:
#         inter_token = re.sub('[,.:%_\+\-\*\/\%_]', '', token)
#         if represents_float(inter_token):
#             new_tokens.append('number')
#         else:
#             new_tokens.append(token)
#     return ' '.join(new_tokens)

def replace_numbers(tweet):
    new_tokens = []
    tokens = tweet.split(' ')
    for token in tokens:
        intermed_token = re.sub('[,.:%_\+\-\*\/\%_]', '', token)
        if represents_float(intermed_token):
            new_tokens.append('number')
        else:
            if not re.findall('[0-9]', token):
                new_tokens.append(token)
    return ' '.join(new_tokens)

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

def replace_punctuation(tweet):
    tokenizer = RegexpTokenizer(r'\w+')
    return " ".join(tokenizer.tokenize(tweet))

def remove_punctuation(text):
    ''' function that is executed after 'replace_emoji' to remove all the uneccesary punctuations'''
    return ' '.join([element for element in text.split(' ') if element not in string.punctuation])

def replace_exclamation(tweet):
    return re.sub('!((\s)?!)+', ' !! ', tweet)

def reduce_white_spaces(tweet):
    return re.sub('( )+', ' ', tweet).strip()

# def pos_tag(tweet):
#     tweet = nltk.word_tokenize(tweet.lower())
#     return nltk.pos_tag(tweet)

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

def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def add_sentiment(tweet):
    ''' function that adds the word 'positive' or 'negative' to the tweet
        depending on the sentiment of the word, obtained from WordNetLemmatizer

        this is a way of giving emphasis to the sentiment of a tweet 
    ''' 
    new_tweet = []  
    tagged_tweet = pos_tag(word_tokenize(tweet))

    for word, tag in tagged_tweet:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            new_tweet.append(word)
            continue

        lemma = WordNetLemmatizer().lemmatize(word, pos=wn_tag)
        if not lemma:
            new_tweet.append(word)
            continue

        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            new_tweet.append(word)
            continue

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        if swn_synset.pos_score() > 0.5:
            new_tweet.append('positive ' + word)
        elif swn_synset.neg_score() > 0.5:
            new_tweet.append('negative ' + word)
        else:
            new_tweet.append(word)
        
    return ' '.join(new_tweet)

#Copyright (c) http://stackoverflow.com/users/1515832/generic-human (http://stackoverflow.com/a/11642687)

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open(DATA_UTILS + 'words-by-frequency.txt').read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)


def split_hashtag_to_words(hashtag):
    try: return infer_spaces(hashtag[1:]).strip()
    except: return hashtag[1:]

def infer_spaces(s):
    ''' Uses dynamic programming to infer the location of spaces in a string
        without spaces.

        Find the best match for the i first characters, assuming cost has
        been built for the i-1 first characters.
        Returns a pair (match_cost, match_length).
    '''

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
    ''' function that is used to separate the '#' from the words
        inside this function there is an invocation of split_hashtag_to_words 
        
        eg. 
    '''
    new_tokens = []
    tokens = tweet.split(' ')
    for token in tokens:
        if token.startswith('#'): new_tokens.append(
            'hashtag ' + split_hashtag_to_words(token))
        else: new_tokens.append(token)
    return ' '.join(new_tokens)
