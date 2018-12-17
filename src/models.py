#file that contains all the models that were used to reach the best one
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Embedding, Activation, Flatten
from keras.optimizers import Adam, SGD


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




def get_estimator(model, input_shape=None):
    ''' provided the model that we want as parameter,
        this function returns the corresponding model with the best possible parameters
    '''
    if model == 'LR':
        return LogisticRegression(max_iter=1000, solver='sag')
    elif model == 'RF':
        return RandomForestClassifier(n_estimators=100, max_depth=50)
    elif model == 'NB':
        return MultinomialNB()
    elif model == 'SVM':
        return SVC(probability=True)
    elif model == 'NN':
        return create_model_NN('NN', input_shape)
    else:
        raise ValueError('Invalid value ML algorithm')
        
def create_model_NN(model_type='NN', input_shape=None):
    ''' create NN model '''
    if model_type == 'NN':
        model = Sequential()
        model.add(Dense(units=64, input_dim=input_shape, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == 'CNN':
        print('TODO CNN')
    return model

def compile_model(model, optimizer='Adam'):
    ''' compile the NN model with the optimizer provided'''
    if optimizer == 'Adam':
        optimizer = Adam(lr=0.03, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    return model
