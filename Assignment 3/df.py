from array import array as py_array
from numpy import arange, array, int8, uint8, zeros, array
from struct import unpack
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
		
def classifier():
	train_images,train_labels = processData('train')
	test_images,test_labels   = processData('test')
	# sss = StratifiedShuffleSplit(train_labels, 3, test_size=0.5, random_state=0)
	# print sss
	# raw_input()
	# for train_index, test_index in sss:
	# 	print train_labels[train_index]
	# 	raw_input()


	train_images, train_labels = refineSets(train_images, train_labels, 1111)
	test_images, test_labels = refineSets(test_images, test_labels, 111)
	
	hog_train_images = [ hog(image) for image in train_images]
	hog_test_images = [ hog(image) for image in test_images]

	print 'Accuracy on test data : '
		#DistanceMetric.get_metric(metric)

	forest_sizes = [100,200,300,400,500]
	for size in forest_sizes:
		clf = RandomForestClassifier(n_estimators=size,criterion='entropy',n_jobs=-1)
		clf.fit(hog_train_images,train_labels)

		print size,"->",clf.score(hog_test_images,test_labels)

classifier()