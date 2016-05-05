from array import array as py_array
from numpy import arange, array, int8, uint8, zeros
from struct import unpack
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
import math


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

def adaBoost(train_hogFeatures,test_hogFeatures,val_hogFeatures,train_labels,val_labels,test_labels):
    pass

def classifier():
    train_images,train_labels = processData('train')
    test_images,test_labels   = processData('test')

    train_images, train_labels = refineSets(train_images, train_labels, 1111)
    test_images, test_labels = refineSets(test_images, test_labels, 111)
    print "Refined--------------------"
    # test, train shortened
    # cross validation on train-set
    val_images, val_labels = refineSets(train_images[20000:],1111)

    hog_train_images = [ hog(image) for image in train_images]
    hog_test_images = [ hog(image) for image in test_images]
    hog_val_images = [hog(image) for image in val_images]
    ntrees=[2,5,10,20,40,100,200,500,800,1000]
    # ntrees=[500]
    max_depth=[2,3]
    # max_depth=[17]
    for i in max_depth:
        for j in ntrees:
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i),n_estimators=j)
            clf.fit(hog_train_images,train_labels)
            print "n_estimators\t"+str(j)+"\t Max_depth\t"+str(i)+"\t Score:" +str(clf.score(hog_val_images,val_labels))

classifier()
         
