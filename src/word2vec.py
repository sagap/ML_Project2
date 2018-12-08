#word2vec implementation
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

def return_mean_vectorizer(tweets, vector_dim=300, min_count=2, epochs=5):
	''' 
	@param tweets list of tweets
	@param vector_dim dimension of word vectors
	@param min_count minimum word frequency
	@param epochs iterations of tweets set
	@return list of vectors, each element of the list corresponds to a tweet, 
			preserving the order of the tweets set
			e.g. the first vector of the list matches the first tweet etc
	'''
	tokens = [tokenizer.tokenize(tweet) for tweet in tweets]
	model = Word2Vec(size=vector_dim, min_count=min_count)
	model.build_vocab(tokens)
	model.train(tokens, epochs=epochs, total_examples=model.corpus_count)
	return np.array([np.mean([model.wv.__getitem__(word) for word in token if word in model.wv] or [np.zeros(vector_dim)], axis=0)
			 for token in tokens])
''' 
Accuracy: 0.7358 (+/- 0.0060) min count = 1 and D = 200, epochs = 2
Accuracy: 0.7341 (+/- 0.0060) min count = 2 and D = 200, epochs = 2
Accuracy: 0.7337 (+/- 0.0046) min count = 2 , D = 300, epochs = 2
Accuracy: 0.7357 (+/- 0.0054) min count = 1  , D = 300, epochs = 2
Accuracy: 0.7100 (+/- 0.0054) min count = 1  , D = 300, epochs = 1
Accuracy: 0.7348 (+/- 0.0054) min count = 1  , D = 400, epochs = 2
Accuracy: 0.7324 (+/- 0.0050) min count = 1 , D = 500 , epochs = 2
Accuracy: 0.7540 (+/- 0.0055) min_count = 1, D = 300, epochs = 5
Accuracy: 0.7568 (+/- 0.0045) min_count = 2, D = 300, epochs = 5
Accuracy: 0.7611 (+/- 0.0052) 2, 300 , 5
Accuracy: 0.7647 (+/- 0.0039) 3, 300 , 5
Accuracy: 0.7772 (+/- 0.0031) 4, 300, 8
'''
# Below... Doc2Vec implementation
from gensim.models import Doc2Vec

''' 
	DBoW
	predicts a probability distribution of words in a paragraph,
	given a randomly-sampled word from the paragraph.
'''


