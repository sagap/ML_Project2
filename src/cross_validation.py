import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold as cross_validation_KFold

def cross_validation(classifier, n_splits=4, ):
	''' function that provided a classifier and the parameters needed,
		performs k-fold cross validation 
	'''
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
	scores = cross_val_score(clf, X, y, cv=kf)
	print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
	return scores