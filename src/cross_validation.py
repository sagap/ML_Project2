import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import helpers

def cross_validation(model, X_train, y , kf, n_splits=4, n_jobs=2):
    ''' function that provided a classifier and the parameters needed,
    performs k-fold cross validation 
    '''
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    scores = cross_val_score(model, X_train, y, cv=kf, n_jobs=n_jobs)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    return scores

def cross_validation_NN(model, X, y):
    splits = np.random.random_integers(4, 6)
    kfold = KFold(n_splits=splits, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        result = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=2, 
                  callbacks=helpers.checkpointing())
    helpers.print_history(result)
    return result