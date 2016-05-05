from os import listdir
from os.path import isfile, join
import sys
import numpy
import cPickle as pickle
import collections, re
import scipy.sparse
from sklearn import svm
import numpy
from sklearn import cross_validation
from sklearn.cross_validation import KFold

def readFile(filename):
	X =[]
	Y=[]
	line=[]
	output =1
	with open(filename) as f:
		line = f.readlines()
	for l in line:
		l = l.strip()
		temp = numpy.zeros(42*3)
		text = l.split(',')
		for i in range(0,42):		
			if text[i] == 'o':
				temp[3*i] = 1
			elif text[i] == 'b':
				temp[3*i+1] = 1
			elif text[i] == 'x':
				temp[3*i+2]=1
		if text[42] == "win":
			output =1
		elif text[42]=="draw":
			output =0
		elif text[42] == "loss":
			output = -1
		# temp has the test and output the label
		X.append(temp)
		Y.append(output)
	return X,Y

def relabel(Y,include):
	# if include =1 , rest = 0
	# if include = -1, make -1 to 1, rest to 0
	# if include = 0, make it 1 rest to 0
	exclude =-99
	Z = []
	for i in range(0,len(Y)):
		# print Y[i], include
		if Y[i]!= include:
			Z.append(0)
		else:
			Z.append(1)
	return numpy.array(Z)

def makeSets(X,A,train, test):
	return	numpy.array(X[train]), numpy.array(X[test]), numpy.array(A[train]), numpy.array(A[test])

def Predict(A_probability,B_probability,C_probability):
	# A - win
	# B - draw
	# C - loss
	n = len(A_probability)
	Z = []
	for i in range(0,n):
		win = (A_probability[i])
		draw = (B_probability[i])
		loss = (C_probability[i])


		if win >= max(draw, loss):
			Z.append(1)
		elif loss >= max(win,draw):
			Z.append(-1)
		else:
			Z.append(0)

	return numpy.array(Z)




def OneVsRest(X,Y,nfolds=5):
	# 1 versus rest
	# win - vs all
	Z = Y
	A_predict =[]
	B_predict = []
	C_predict = []
	A = relabel(Y,1)
	totalAccc =0
	# 1 = win, 0 = others
	B = relabel(Y,0)
	# 1 = draw , 0 = others

	C = relabel(Y,-1)
	# 1 = loss, 0 = others
	kf = KFold(len(Y), n_folds=nfolds)
	for train, test in kf:
		X_train, X_test, Y_train, Y_test = makeSets(X,A,train, test)

		clf = svm.LinearSVC()
		clf.fit(X_train, Y_train) 
		A_probability = clf.decision_function(X_test)
		print "A done"
		# raw_input()
		X_train, X_test, Y_train, Y_test = makeSets(X,B,train, test)
		clf = svm.LinearSVC()
		clf.fit(X_train, Y_train)
		B_probability = clf.decision_function(X_test)
		print "B done" 

		X_train, X_test, Y_train, Y_test = makeSets(X,C,train, test)
		clf = svm.LinearSVC()
		clf.fit(X_train, Y_train) 
		C_probability = clf.decision_function(X_test)
		print "C done"
		Y_predicted = Predict(A_probability,B_probability,C_probability) 

		# raw_input()
		accuracy =0
		for i in range(0, len(Y[test])):
			if Y_predicted[i] == Y[test][i]:
				accuracy+=1
		totalAccc += 1.0*accuracy/len(Y_test) 
		print 1.0*accuracy/len(Y_test)
		print "one round done"

	return totalAccc/5

def labelOnevsOne(X,Y, pos,neg):
	X_new=[]
	Y_new=[]
	for i in range(0,len(Y)):
		if Y[i]==pos:
			X_new.append(X[i])
			Y_new.append(1)
		elif Y[i]==neg:
			X_new.append(X[i])
			Y_new.append(0)
	return numpy.array(X_new),numpy.array(Y_new)

def predictOneVsOne(A_predict,B_predict,C_predict,A,B,C):
	Z=[]
	for i in range(0,len(A_predict)):
		if A_predict[i] == 1 and C_predict[i] == 0:
			Z.append(1)
		elif B_predict[i] == 1 and A_predict[i] ==0:
			Z.append(0)
		elif C_predict[i] == 1 and B_predict[i] == 0:
			Z.append(-1)
		else:
			# print A_predict[i], B_predict[i], C_predict[i]
			# print A[i], B[i],C[i]
			if A[i] >= max(B[i],C[i]):
				Z.append(1)
			elif B[i] >= max(A[i],C[i]):
				Z.append(0)
			elif C[i] >= max(A[i],B[i]):
				Z.append(-1)
			else:
				print "Not sure what to do"
				raw_input()
	return numpy.array(Z)




def OneVsOne(X,Y, nfolds):
	Z = Y
	A_predict =[]
	B_predict = []
	C_predict = []
	

	totalAccc =0

	kf = KFold(len(Y), n_folds=nfolds)
	for train, test in kf:
		# train X, train Y
		X_A, Y_A = labelOnevsOne(numpy.array(X[train]),numpy.array(Y[train]),1,0)
		X_B, Y_B = labelOnevsOne(numpy.array(X[train]),numpy.array(Y[train]),0,-1)
		X_C, Y_C = labelOnevsOne(numpy.array(X[train]),numpy.array(Y[train]),-1,1)

		X_test = numpy.array(X[test])
		Y_test = numpy.array(Y[test])
		clf = svm.LinearSVC()
		clf.fit(X_A, Y_A) 
		A_predict = clf.predict(X_test)
		A_probability = clf.decision_function(X_test)
		print "A done"
		# raw_input()
		
		clf = svm.LinearSVC()
		clf.fit(X_B, Y_B)
		B_predict = clf.predict(X_test)
		B_probability = clf.decision_function(X_test)
		print "B done" 

		
		clf = svm.LinearSVC()
		clf.fit(X_C, Y_C) 
		C_predict = clf.predict(X_test)
		C_probability = clf.decision_function(X_test)
		print "C done"
		Y_predicted = predictOneVsOne(A_predict,B_predict,C_predict, A_probability,B_probability,C_probability) 

		# raw_input()
		accuracy =0
		for i in range(0, len(Y[test])):
			if Y_predicted[i] == Y[test][i]:
				accuracy+=1
		totalAccc += 1.0*accuracy/len(Y_test) 
		print 1.0*accuracy/len(Y_test)
		print "one round done"

	return totalAccc/5


if __name__ == "__main__":
	X,Y = readFile("connect-4.data")
	accuracy=OneVsRest(numpy.array(X),numpy.array(Y),5)
	print "Final accuracy One vs Rest", accuracy
	accuracy=OneVsOne(numpy.array(X),numpy.array(Y),5)
	print "Final accuracy One-vs-One", accuracy




# clf = svm.SVC(kernel='linear', C=1)
# >>> scores = cross_validation.cross_val_score(
# ...    clf, iris.data, iris.target, cv=5))