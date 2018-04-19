#import tensorflow as tf
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import random
#from lex import lexer, cpp_keywords
import lex
import itertools
from collections import OrderedDict

import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold

def get_freqs(full_filename_path):
    #print(full_filename_path)
    source = open(full_filename_path, 'r').read()
    ordered = OrderedDict(lex.get_freqs_from_cpp_source(source))
    #tmp = lex.get_ngram(1, source)
    #print(tmp)
    #ordered = OrderedDict(tmp)
    #print(ordered)
    #print("---------------")
    return list(ordered.values())

def get_ngrams(full_filename_path, n):
    source = open(full_filename_path, 'r').read()
    tmp = lex.get_ngram(n, source)
    ordered = OrderedDict(tmp)
    return list(ordered.values())

TRAINING_PART = 0.75

#path = "spb-2017/"
#path = "north-2015-runs/"
path = "north-2011-runs/"

onlyfiles = [ f for f in listdir(path) if isfile(join(path, f)) ]

cppfiles = list(filter(lambda x : x[-3:] == "cpp", onlyfiles))
#cppfiles = cppfiles[:2] #filter
file_to_task = {}
for f in cppfiles:
    file_to_task[f] = f[4]

data_len = len(cppfiles)
random.shuffle(cppfiles)
if TRAINING_PART < 1:
    training_data, test_data = train_test_split(cppfiles, train_size=TRAINING_PART)
else:
    training_data = cppfiles
    test_data = []

#training_data = cppfiles[:int(len(cppfiles)*TRAINING_PART)]
#test_data = cppfiles[-data_len+len(training_data):]

print(len(training_data))

print(len(test_data))

def get_features(path):
    return get_ngrams(path, 1)
    #return get_freqs(path)

X = []
y = []
print("Training data")
for f in training_data:
    print(f)
    X.append(get_features(path + f))
    y.append(f[4])

X_test = []
y_test = []

print("Test data")
for f in test_data:
    print(f)
    X_test.append(get_features(path + f))
    y_test.append(f[4])

scaler = StandardScaler()
scaler.fit(X + X_test)

#scaler = StandardScaler()
#scaler.fit(X_test)

X = scaler.transform(X)

print(X)

if len(X_test) > 0:
    X_test = scaler.transform(X_test)
    #print(X_test)

sel = VarianceThreshold()
if len(X_test) > 0:
    sel.fit(np.concatenate((X, X_test)))
else:
    sel.fit(X, X_test)

X = sel.transform(X)
print(X.shape)
if len(X_test) > 0:
    X_test = sel.transform(X_test)

names = [ #"Linear SVM", "Decision Tree",
        "Random Forest", "Neural Net lbfgs",
        #"Neural Net adam", "Neural Net sgd"
        ]

classifiers = [
        SVC(kernel="linear", C=0.025),
        DecisionTreeClassifier(),
        RandomForestClassifier(max_features=None),
        MLPClassifier(solver="lbfgs", max_iter=5000, alpha=0.3),
        MLPClassifier(solver="adam", max_iter=5000, alpha=0.3),
        MLPClassifier(solver="sgd", max_iter=5000, alpha=0.3),
    ]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def calc_cnf_matrix(classifier, name, X, y, X_test, y_test):
    classifier.fit(X, y)
    y_pred = classifier.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, title=name, classes=clf.classes_, normalize=True)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, title=name, classes=clf.classes_, normalize=False)

def calc_cross_vadil(classifier, name, X, y, cv):
    score = cross_val_score(classifier, X, y, cv=cv)
    print(name + ": " + str(np.average(score)) + " -> " +  str(score))

def calc_score(classifier, name, X, y, X_test, y_test):
    classifier.fit(X, y)
    score = classifier.score(X_test, y_test)
    print(name + ": " + str(score))

for name, clf in zip(names, classifiers):
    calc_cnf_matrix(clf, name, X, y, X_test, y_test)
    #calc_score(clf, name, X, y, X_test, y_test)
    #calc_cross_vadil(clf, name, X, y, cv=5)

    #clf.fit(X,y)
    #score = cross_val_score(clf, X, y, cv=5)
    #y_pred = clf.predict(X_test)
    #print(y_test)
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #np.set_printoptions(precision=2)
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, title=name, classes=clf.classes_, normalize=True)
plt.show()

    #score = clf.score(X_test, y_test)
    #print(name + ": " + str(np.average(score)) + " -> " +  str(score))

#count_right = 0
#for f in test_data:
#    #prediction = clf.predict([ get_freqs(path + test_data[0]) ])
#    score = clf.score(X_test, y_test)
#    print(score)
#    #if prediction[0] == test_data[0][4]:
#    #    count_right += 1
#
#print(count_right / len(test_data))