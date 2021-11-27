import numpy as np
import pandas as pd
from sklearn import svm
import math

def linear_kernel(train_set,test_set):
    y_train = train_set['label']
    x_train = train_set.drop(columns = ['label'])    
    y_test = test_set['label']
    x_test = test_set.drop(columns = ['label'])

    clf = svm.SVC(kernel='linear')
    clf.fit(x_train,y_train)
    preds = clf.predict(x_test)
    accuracy = np.mean(preds == y_test)
    n_support_vectors = len(clf.support_vectors_)

    return accuracy, n_support_vectors

def poly_kernel(train_set,test_set, C, Q):
    y_train = train_set['label']
    x_train = train_set.drop(columns = ['label'])    
    y_test = test_set['label']
    x_test = test_set.drop(columns = ['label'])

    clf = svm.SVC(kernel='poly', C = C, degree = Q, coef0=1, gamma=1)
    score = clf.fit(x_train,y_train).score(x_train, y_train)
    preds = clf.predict(x_test)
    accuracy = np.mean(preds == y_test)
    n_support_vectors = len(clf.support_vectors_)

    return 1 - score, 1-accuracy, n_support_vectors

def rbf_kernel(train_set,test_set, C):
    y_train = train_set['label']
    x_train = train_set.drop(columns = ['label'])    
    y_test = test_set['label']
    x_test = test_set.drop(columns = ['label'])

    clf = svm.SVC(kernel='rbf', C=C, gamma=1)
    score = clf.fit(x_train,y_train).score(x_train, y_train)
    preds = clf.predict(x_test)
    accuracy = np.mean(preds == y_test)

    return 1-score, 1-accuracy

if __name__ == '__main__':
    train = pd.read_fwf('features.train', header = None, names = ['label', 'feature_1', 'feature_2'])
    test = pd.read_fwf('features.test', header = None, names = ['label', 'feature_1', 'feature_2'])
    
    # print(train.head())
    # print(test.head())

    # print(train.describe())
    # print(test.describe())
    
    reqd_labels = [1.0,5.0]
    train = train[train['label'].isin(reqd_labels)]
    test = test[test['label'].isin(reqd_labels)]

    accuracy, n_support_vectors = linear_kernel(train, test)

    print("Linear Kernel:")
    print("Accuracy with linear kernel is %f and number of support vectors are %d" %(accuracy, n_support_vectors))
    
    for idx in [50, 100, 200, 800]:
        accuracy, n_support_vectors = linear_kernel(train[:idx], test)
        print("Accuracy with linear kernel trained on first %d points is %f and number of support vectors are %d" %(idx, accuracy, n_support_vectors))


    print("\n\n Polynomial Kernel")
    for c in [0.0001, 0.001, 0.01, 1]:
        for q in [2,5]:
            training_error, test_error, support_vectors = poly_kernel(train, test, c, q)
            print("At C=%f and Q=%d, training error: %f\t test error: %f\t support vectors: %d" %(c, q, training_error, test_error, support_vectors))

    print("\n\n RBF Kernel")
    for c in[0.01, 1, 100, math.pow(10,4), math.pow(10, 6)]:
        train_error, test_error = rbf_kernel(train, test, c)
        print("For C=%d, training error is %f and test error is %f" %(c, train_error, test_error))