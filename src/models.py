#file that contains all the models that were used to reach the best one
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

def logistic_regression_count_vectorizer(lines, threshold_pos_neg):
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(lines)

	y = np.zeros(shape=(len(lines)))
	y[:len(threshold_pos_negthreshold_pos_neg)] = 1
	y[len(threshold_pos_neg):] = -1

	clf = LogisticRegression().fit(X, y)
	kf = KFold(n_splits=5, shuffle=True, random_state=0)
	scores = cross_val_score(clf, X, y, cv=kf)
	print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
