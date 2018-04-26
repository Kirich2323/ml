#import tensorflow as tf
import matplotlib.pyplot as plt
#import random
import lex
import itertools
import feature_extractor
import label_extractor
import numpy as np

from matplotlib.colors import ListedColormap
#from sklearn.model_selection import train_test_split
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
from sklearn.base import clone

classifier_names = [ 
      "Linear SVM",
      "Decision Tree",
      "Random Forest",
      "Neural Net lbfgs",
      "Neural Net adam",
      "Neural Net sgd",
      "KNeighborsClassifier",
      "NN 2 hidden layers", 
    ]

classifiers = { "Linear SVM" : SVC(kernel="linear", C=0.025),
        "Decision Tree" : DecisionTreeClassifier(),
        "Random Forest" : RandomForestClassifier(max_features=None),
        "Neural Net lbfgs" : MLPClassifier(solver="lbfgs", max_iter=5000, alpha=0.3),
        "Neural Net adam" : MLPClassifier(solver="adam", max_iter=5000, alpha=0.3),
        "Neural Net sgd" : MLPClassifier(solver="sgd", max_iter=5000, alpha=0.3),
        "NN 2 hidden layers" : MLPClassifier(hidden_layer_sizes=(50, 50), solver="lbfgs", max_iter=5000, alpha=0.3),
        "KNeighborsClassifier" : KNeighborsClassifier()
}

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

    #print(cm)

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
    score = cross_val_score(classifier, X, y, cv=cv, n_jobs=-1)
    print(name + ": " + str(np.average(score)) + " -> " +  str(score))
    return score

def calc_score(classifier, name, X, y, X_test, y_test):
    classifier.fit(X, y)
    score = classifier.score(X_test, y_test)
    #print(name + ": " + str(score))
    return score

def get_freqs(full_filename_path):
    source = open(full_filename_path, 'r').read()
    ordered = OrderedDict(lex.get_freqs_from_cpp_source(source))
    return list(ordered.values())

def get_ngrams(full_filename_path, n):
    source = open(full_filename_path, 'r').read()
    tmp = lex.get_ngram(n, source)
    ordered = OrderedDict(tmp)
    return list(ordered.values())


TRAINING_PART = 0.5

#path = "spb-2017/"
#path = "north-2015-runs/"
path = "north-2011-runs/"
#path = "neerc-2014-runs/"
#path = "neerc-2016-runs/"

#xml = "north-2011-log/log.xml"
N = 10


scores = np.zeros([len(classifier_names), N])
print(scores)
for i in range(N):
    #feature_ext = feature_extractor.NgramExtractor(path_to_sources=path, training_part=TRAINING_PART, n=1,
    #                                               scale_features=True, variance_threshold=False)
    feature_ext = feature_extractor.Word2VecExtractor(path_to_sources=path, training_part=TRAINING_PART, size=50)

    label_ext = label_extractor.ProblemExtractor()
    #label_ext = label_extractor.VerdictExtractor(xml=xml)
    y = label_ext.get_labels(feature_ext.training_data) #todo remove data into separate class from feature_extractor
    y_test = label_ext.get_labels(feature_ext.test_data)
    j = 0
    for name in classifier_names:
        clf = clone(classifiers[name])
        #calc_cnf_matrix(clf, name, feature_ext.X, y, feature_ext.X_test, y_test)
        score = calc_score(clf, name, feature_ext.X, y, feature_ext.X_test, y_test)
        scores[j][i] = score
        #calc_cross_vadil(clf, name, feature_ext.X, y, cv=5)
        j += 1
print(scores)
means = scores.mean(1)
std = scores.std(1)
mins = scores.min(1)
maxs = scores.max(1)
#print(means)
#print(std)
#print(mins)
#print(maxs)
#print(len(scores))
plt.errorbar(range(len(scores)), means, std, fmt='ok', lw=3)
plt.errorbar(range(len(scores)), means, [means - mins, maxs - means],
             fmt='.k', ecolor='gray', lw=1)
plt.xticks(range(len(scores)), classifier_names, rotation=30)
plt.show()