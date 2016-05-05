from array import array as py_array
from numpy import arange, array, int8, uint8, zeros, array
from struct import unpack
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
import cv2
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

from time import time
from operator import itemgetter
from scipy.stats import randint

import numpy as np
def processData(prefix):
    
    digits = arange(10)
    dire = ''

    f = open(dire + prefix +'-images.idx3-ubyte', 'rb')
    magic_number, nb_digits, nb_rows, nb_cols = unpack(">IIII", f.read(16))
    nb_pixels = nb_rows * nb_cols
    images = array(py_array("B", f.read()))
    f.close()

    f = open(dire + prefix +'-labels.idx1-ubyte', 'rb')
    magic_number, nb_digits = unpack(">II", f.read(8))
    labels = array(py_array("b", f.read()))
    f.close()

    indices = [i for i in range(nb_digits) if labels[i] in digits]  #just asserting
    nb_digits = len(indices)

    my_images = zeros((nb_digits, nb_rows,nb_cols), dtype=uint8)
    my_labels = zeros((nb_digits,), dtype=int8)
    
    for i in range(nb_digits):
        index = indices[i]
        for row in xrange(nb_rows):
            my_images[i][row] = array(images[(index*nb_pixels + row*nb_cols):(index*nb_pixels +  (row+1)*nb_cols)])

        my_labels[i] = labels[index]

    return my_images,my_labels

def refineSets(images, labels, size):
	result = zeros(10)
	count =0
	X=[]
	Y=[]

	for i in range(0,len(labels)): 
		label = labels[i]
		feature = images[i]
		if result[label]<=size:
			X.append(feature)
			Y.append(label)
			count+=1
			result[label]+=1
		if count > size*10:
			break

	return array(X), array(Y)
def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters
def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return  top_params

def classifier():
	train_images,train_labels = processData('train')
	test_images,test_labels   = processData('test')

	train_images, train_labels = refineSets(train_images, train_labels, 1111)
	test_images, test_labels = refineSets(test_images, test_labels, 111)
	
	hog_train_images = [ hog(image) for image in train_images]
	hog_test_images = [ hog(image) for image in test_images]

	print 'Accuracy on test data : '
	#DistanceMetric.get_metric(metric)
	param_grid = {
			"criterion": ["gini", "entropy"],
			"min_samples_split": [2, 10, 20],
			"max_depth": [None, 2, 5, 10],
			"min_samples_leaf": [1, 5, 10],
		}

	clf = DecisionTreeClassifier(min_samples_split=2, criterion='gini', max_depth=None, min_samples_leaf= 5)
	# ts_gs = run_gridsearch(hog_train_images, train_labels, clf, param_grid, cv=5)
	clf.fit(hog_train_images,train_labels)

	print clf.score(hog_test_images,test_labels)
# 
classifier()