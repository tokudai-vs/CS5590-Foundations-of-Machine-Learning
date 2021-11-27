import numpy as np
import pandas as pd
from sklearn import svm
import math
import numpy as np

def linear_kernel(x_train, y_train, x_test, y_test):
    clf = svm.SVC(kernel='linear')
    score = clf.fit(x_train,y_train).score(x_train, y_train)
    preds = clf.predict(x_test)
    accuracy = np.mean(preds == y_test)
    n_support_vectors = len(clf.support_vectors_)

    return 1-score, 1-accuracy, n_support_vectors

def rbf_kernel(x_train, y_train, x_test, y_test, gamma):
    clf = svm.SVC(kernel='rbf', gamma = gamma)
    score = clf.fit(x_train,y_train).score(x_train, y_train)
    preds = clf.predict(x_test)
    accuracy = np.mean(preds == y_test)
    n_support_vectors = len(clf.support_vectors_)

    return 1-score, 1-accuracy, n_support_vectors

def poly_kernel(x_train, y_train, x_test, y_test, Q, coef):
    clf = svm.SVC(kernel='poly', degree=Q, coef0 = coef, gamma=1)
    score = clf.fit(x_train,y_train).score(x_train, y_train)
    preds = clf.predict(x_test)
    accuracy = np.mean(preds == y_test)
    n_support_vectors = len(clf.support_vectors_)

    return 1-score, 1-accuracy, n_support_vectors

if __name__ == '__main__':
    x_train = np.loadtxt('GISETTE/gisette_train.data')
    y_train = np.loadtxt('GISETTE/gisette_train.labels')
    x_test = np.loadtxt('GISETTE/gisette_valid.data')
    y_test = np.loadtxt('GISETTE/gisette_valid.labels')
    
    train_error, test_error, support_vectors = linear_kernel(x_train, y_train, x_test, y_test)
    print("Linear Kernel: Train error: %f\tTest error: %f\tSupport vectors: %d" % (train_error, test_error, support_vectors))

    train_error, test_error, support_vectors = rbf_kernel(x_train, y_train, x_test, y_test, 0.001)
    print("\nRBF Kernel: Train error: %f\tTest error: %f\tSupport vectors: %d" % (train_error, test_error, support_vectors))

    train_error, test_error, support_vectors = poly_kernel(x_train, y_train, x_test, y_test, 2, 1)
    print("\nPoly Kernel: Train error: %f\tTest error: %f\tSupport vectors: %d" % (train_error, test_error, support_vectors))