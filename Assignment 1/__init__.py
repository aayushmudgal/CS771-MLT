from os import listdir
from os.path import isfile, join
import sys
import numpy
import cPickle as pickle
import collections, re
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
def createCorpus(data,i, binaryX="False", stopWords=None, lemmatize="False", tfidf= "False", useidf="True"):  # will vectorize BOG using frequency as the parameter and will return the required arrays
	X_train =[]
	X_test=[]
	Y_train=[]
	Y_test=[]

	for key in data:
		if key==i:

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


def crossValidation(data, question):

	if question=="2a":
		accuracy=0				# with frequency
		for i in range(1,11):
			X_train, Y_train,X_test,Y_test = createCorpus(data,i, binaryX="False", stopWords=None, lemmatize="False")  # with frequency   

			clf = MultinomialNB().fit(X_train, Y_train)   # naive Bayes
			predicted = clf.predict(X_test)
			turnAccuracy = numpy.mean(predicted==Y_test)
			print "Round \t"+str(i)+"\t accuracy=\t"+str(turnAccuracy)
			accuracy += turnAccuracy
			tp=0
			tn=0
			fp=0
			fn=0
			for k in range(len(predicted)):
				if predicted[k] == 1 and Y_test[k] == 1:
					tp+=1
				elif predicted[k] == 1 and Y_test==0:
					fp+=1
				elif predicted[k]==0 and Y_test==1:
					fn+=1
				elif predicted[k]==0 and Y_test==0:
					tn+=1
			print "Precision \t" , 1.0*tp/(1.0*tp+fp)
			print "Recall \t", 1.0*tp/(1.0*tp+fn) 
		print "Accuracy Naive-Bayes + frequency of words ", accuracy / 10

	elif question=="2ab":
		accuracy=0						# without frequency
		for i in range(1,11):
			X_train, Y_train,X_test,Y_test = createCorpus(data,i, binaryX="True", stopWords=None, lemmatize="False")  # with frequency   
			clf = MultinomialNB().fit(X_train, Y_train)   # naive Bayes
			predicted = clf.predict(X_test)
			turnAccuracy = numpy.mean(predicted==Y_test)
			print "Round \t"+str(i)+"\t accuracy=\t"+str(turnAccuracy)
			accuracy += turnAccuracy
		print "Accuracy Naive-Bayes + not frequency of words ", accuracy / 10

	elif question=="2b":				# removing stop words
		accuracy=0
		for i in range(1,11):
			X_train, Y_train,X_test,Y_test = createCorpus(data,i, binaryX="False", stopWords="english", lemmatize="False")  # with frequency   
			clf = MultinomialNB().fit(X_train, Y_train)   # naive Bayes
			predicted = clf.predict(X_test)
			turnAccuracy = numpy.mean(predicted==Y_test)
			print "Round \t"+str(i)+"\t accuracy=\t"+str(turnAccuracy)

			accuracy += turnAccuracy
		print "Accuracy Naive-Bayes + Stop-words removal ", accuracy / 10

	elif question == "2c":
		accuracy=0
		for i in range(1,11):
			X_train, Y_train,X_test,Y_test = createCorpus(data,i, binaryX="False", stopWords="english", lemmatize="True")  # with frequency   
			clf = MultinomialNB().fit(X_train, Y_train)   # naive Bayes
			predicted = clf.predict(X_test)
			turnAccuracy = numpy.mean(predicted==Y_test)
			print "Round \t"+str(i)+"\t accuracy=\t"+str(turnAccuracy)
			accuracy += turnAccuracy
		print "Accuracy Naive-Bayes + Stop-words removal + Lemmatizing ", accuracy / 10
	elif question == "1a":
		accuracy=0

		for i in range(1,11):

			X_train, Y_train,X_test,Y_test = createCorpus(data,i, binaryX="True", stopWords="english", lemmatize="True")  # with frequency   
			clf = Perceptron(n_jobs=-1)
			X = clf.fit(X_train, Y_train)
			predicted =clf.predict(X_test)
			turnAccuracy = numpy.mean(predicted==Y_test)
			print "Round \t"+str(i)+"\t accuracy=\t"+str(turnAccuracy)
			accuracy += turnAccuracy
		print "Accuracy Perceptron + Stop-words removal + Lemmatizing ", accuracy / 10

	elif question == "1b":
		accuracy =0
		for i in range(1,11):
			X_train, Y_train,X_test,Y_test = createCorpus(data,i, binaryX="True", stopWords="english", lemmatize="True", tfidf="True", useidf=False)  # with frequency   
			clf = Perceptron(n_jobs=-1)
			X = clf.fit(X_train, Y_train)
			predicted =clf.predict(X_test)
			turnAccuracy = numpy.mean(predicted==Y_test)
			print "Round \t"+str(i)+"\t accuracy=\t"+str(turnAccuracy)
			accuracy += turnAccuracy
		print "Accuracy Perceptron + stop-words+ tf+ lemmatizing", accuracy / 10
		# term frequency set 
	elif question == "1c":
		# tf-idf
		accuracy =0
		for i in range(1,11):
			X_train, Y_train,X_test,Y_test = createCorpus(data,i, binaryX="True", stopWords="english", lemmatize="True", tfidf="True")  # with frequency   
			clf = Perceptron(n_jobs=-1)
			X = clf.fit(X_train, Y_train)
			predicted =clf.predict(X_test)
			turnAccuracy = numpy.mean(predicted==Y_test)
			print "Round \t"+str(i)+"\t accuracy=\t"+str(turnAccuracy)
			accuracy += turnAccuracy
			tp=0
			tn=0
			fp=0
			fn=0
			for k in range(len(predicted)):
				if predicted[k] == 1 and Y_test[k] == 1:
					tp+=1
				elif predicted[k] == 1 and Y_test==0:
					fp+=1
				elif predicted[k]==0 and Y_test==1:
					fn+=1
				elif predicted[k]==0 and Y_test==0:
					tn+=1
			print "Precision \t" , 1.0*tp/(1.0*tp+fp)
			print "Recall \t", 1.0*tp/(1.0*tp+fn) 			
		print "Accuracy Perceptron + stop-words+ tfidf+ lemmatizing", accuracy / 10
	elif question == "3a":
		accuracy=0
		for i in range(1,11):

			X_train, Y_train,X_test,Y_test = createCorpus(data,i, binaryX="True", stopWords="english", lemmatize="True")  # with frequency   
			clf = LDA()
			X = clf.fit(X_train.todense(), Y_train)
			predicted =clf.predict(X_test.todense())
			turnAccuracy = numpy.mean(predicted==Y_test)
			print "Round \t"+str(i)+"\t accuracy=\t"+str(turnAccuracy)
			accuracy += turnAccuracy
		print "Accuracy LDA + Stop-words removal + Lemmatizing ", accuracy / 10

	elif question == "3b":
		accuracy =0
		for i in range(1,11):
			X_train, Y_train,X_test,Y_test = createCorpus(data,i, binaryX="True", stopWords="english", lemmatize="True", tfidf="True", useidf=False)  # with frequency   
			clf = LDA()
			X = clf.fit(X_train.todense(), Y_train)
			predicted =clf.predict(X_test.todense())
			turnAccuracy = numpy.mean(predicted==Y_test)
			print "Round \t"+str(i)+"\t accuracy=\t"+str(turnAccuracy)
			accuracy += turnAccuracy
		print "Accuracy LDA + stop-words+ tf+ lemmatizing", accuracy / 10
		# term frequency set 
	elif question == "3c":
		# tf-idf
		accuracy =0
		for i in range(1,11):
			X_train, Y_train,X_test,Y_test = createCorpus(data,i, binaryX="True", stopWords="english", lemmatize="True", tfidf="True")  # with frequency   
			clf = LDA()
			X = clf.fit(X_train.todense(), Y_train)
			predicted =clf.predict(X_test.todense())
			turnAccuracy = numpy.mean(predicted==Y_test)
			print "Round \t"+str(i)+"\t accuracy=\t"+str(turnAccuracy)
			accuracy += turnAccuracy
		print "Accuracy LDA + stop-words+ tfidf+ lemmatizing", accuracy / 10




if __name__ == "__main__":
	loadedData=pickle.load( open( "loadedData", "rb" ) )
	questions = ["3a","3b","3c", "2ab"]
	# questions = ["1a"]
	#for q in questions:
	#	print "---------------------------"
	crossValidation(loadedData, "3b")
	#	print "--------------------------"

	
