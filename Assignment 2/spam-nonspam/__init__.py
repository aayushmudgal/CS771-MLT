from os import listdir
from os.path import isfile, join
import sys
import numpy
import cPickle as pickle
import collections, re
import scipy.sparse
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.lda import LDA
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pylab as pl

def createCorpus(data,i, binaryX="False", stopWords=None, lemmatize="False", tfidf= "False", useidf="True"):  # will vectorize BOG using frequency as the parameter and will return the required arrays
	X_train =[]
	X_test=[]
	Y_train=[]
	Y_test=[]

	for key in data:
		if key in i:

			for filename in data[key]:
				text = data[key][filename][0]
				if lemmatize == "True":
					port =  WordNetLemmatizer()
					text = " ".join([port.lemmatize(k,"v") for k in text.split()])
				X_test.append(text)
				Y_test.append(data[key][filename][1])
		else:
			for filename in data[key]:
				text = data[key][filename][0]
				if lemmatize == "True":
					port =  WordNetLemmatizer()
					text = " ".join([port.lemmatize(k,"v") for k in text.split()])
				X_train.append(text)
				Y_train.append(data[key][filename][1])
	if tfidf == "False":
		vectorizer = CountVectorizer(min_df=1, binary= binaryX, stop_words=stopWords)
		X_train_ans = vectorizer.fit_transform(X_train)
		X_test_ans = vectorizer.transform(X_test)
		return X_train_ans, Y_train, X_test_ans,Y_test
	elif tfidf == "True":
		vectorizer = TfidfVectorizer(min_df=1, use_idf=useidf)
		X_train_ans = vectorizer.fit_transform(X_train)
		X_test_ans = vectorizer.transform(X_test)

		return X_train_ans, Y_train, X_test_ans,Y_test

def showconfusionmatrix(cm, typeModel):
    pl.matshow(cm)
    pl.title('Confusion matrix for '+typeModel)
    pl.colorbar()
    pl.show()
def crossValidation(data):

	accuracy=0				# with frequency
	for i in [1]:
		testSet = [2*i+1,2*i+2]
		X_train, Y_train,X_test,Y_test = createCorpus(data,testSet, binaryX="False", stopWords="english", lemmatize="False") 
		print len(Y_train)+len(Y_test)
		count=0
		for i in range(0,len(Y_train)):
			if Y_train[i]==1:
				count+=1
		for i in range(0,len(Y_test)):
			if Y_test[i] ==1:
				count+=1
		print "count",count
		raw_input()
		C=4
		svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
		rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, Y_train)
		poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, Y_train)
		lin_svc = svm.LinearSVC(C=C).fit(X_train, Y_train)
		titles = ['SVC with linear kernel',
			'LinearSVC (linear kernel)',
			'SVC with RBF kernel',
			'SVC with polynomial (degree 3) kernel']
		filename=["svc.png", "LinearSVC.png", "rbf.png", "poly.png"]
		for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
			print "Accuracy \t +"+titles[i]+str(clf.score(X_test,Y_test))
			predicted = clf.predict(X_test)
			cm = confusion_matrix(predicted, Y_test)
			showconfusionmatrix(cm, titles[i])
			

if __name__ == "__main__":
	loadedData=pickle.load( open( "loadedData", "rb" ) )

	crossValidation(loadedData)
	#	print "--------------------------"

# Accuracy  0.991011816554  C 2
# Accuracy  0.991011816554  C 4
# Accuracy  0.991011816554  C 10
# Accuracy  0.991011816554  C 12
# Accuracy  0.991011816554  C 100

# Accuracy 	 +SVC with linear kernel0.987889273356
# Accuracy 	 +LinearSVC (linear kernel)0.987889273356
# Accuracy 	 +SVC with RBF kernel0.837370242215
# Accuracy 	 +SVC with polynomial (degree 3) kernel0.833910034602

# Accuracy 	 +SVC with linear kernel0.989619377163
# Accuracy 	 +LinearSVC (linear kernel)0.989619377163
# Accuracy 	 +SVC with RBF kernel0.840830449827
# Accuracy 	 +SVC with polynomial (degree 3) kernel0.833910034602
