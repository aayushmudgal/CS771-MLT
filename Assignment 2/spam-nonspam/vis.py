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

def createCorpus(data,i):  # will vectorize BOG using frequency as the parameter and will return the required arrays
	X_train =[]
	X_test=[]
	Y_train=[]
	Y_test=[]
	for key in data:
		if key in i:
			for filename in data[key]:
				text = data[key][filename][0]
				port =  WordNetLemmatizer()
				text = " ".join([port.lemmatize(k,"v") for k in text.split()])
				X_test.append(text)
				Y_test.append(data[key][filename][1])
		else:
			for filename in data[key]:
				text = data[key][filename][0]
				port =  WordNetLemmatizer()
				text = " ".join([port.lemmatize(k,"v") for k in text.split()])
				X_train.append(text)
				Y_train.append(data[key][filename][1])
		vectorizer = CountVectorizer(min_df=1, binary= False, stop_words="english")
		X_train_ans = vectorizer.fit_transform(X_train)
		X_test_ans = vectorizer.transform(X_test)
		return X_train_ans, Y_train, X_test_ans,Y_test


def crossValidation(data):

	accuracy=0				# with frequency
	for i in [0]:
		testSet = [2*i+1,2*i+2]
		X_train, Y_train,X_test,Y_test = createCorpus(data,testSet)  # with frequency   
		num_samples_to_plot = 1000
		X_train_temp, Y_train_temp = shuffle(X_train, Y_train)
		X_train_temp, Y_train_temp = X_train[:num_samples_to_plot], Y_train[:num_samples_to_plot] 		

		pca = PCA(n_components=2)
		
		
		X = pca.fit_transform(X_train_temp.toarray())
		print X.shape
		y=Y_train_temp
		h = .02  # step size in the mesh
		# Create color maps
		cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
		cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
		

		C=4
		svc = svm.SVC(kernel='linear', C=C).fit(X, y)
		rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
		poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
		lin_svc = svm.LinearSVC(C=C).fit(X, y)
		titles = ['SVC with linear kernel',
			'LinearSVC (linear kernel)',
			'SVC with RBF kernel',
			'SVC with polynomial (degree 3) kernel']
		filename=["svc.png", "LinearSVC.png", "rbf.png", "poly.png"]
		for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
			clf1=clf
	
				# we create an instance of Neighbours Classifier and fit the data.

			# Plot the decision boundary. For that, we will assign a color to each
			# point in the mesh [x_min, m_max]x[y_min, y_max].
			x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
			y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
			xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
							numpy.arange(y_min, y_max, h))
			Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
			# Put the result into a color plot
			Z = Z.reshape(xx.shape)
			plt.figure()
			plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

			# Plot also the training points
			plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
			plt.xlim(xx.min(), xx.max())
			plt.ylim(yy.min(), yy.max())
			
			clf1.fit(X_train,Y_train)
			plt.title(titles[i])
			plt.savefig(filename[i])
		

if __name__ == "__main__":
	loadedData=pickle.load( open( "loadedData", "rb" ) )
	crossValidation(loadedData)
	#	print "--------------------------"
