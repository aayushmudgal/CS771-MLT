from array import array as py_array
from numpy import arange, array, int8, uint8, zeros
from os import chmod, remove, rmdir, walk
from os.path import join
from requests import head
from stat import S_IWUSR
from struct import unpack
from tempfile import mkdtemp
from itertools import product
from matplotlib.pyplot import cm, show, subplots
from numpy import array, ceil, floor, nan, set_printoptions, sqrt, zeros
from array import array as py_array
from numpy import arange, array, int8, uint8, zeros
from struct import unpack
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(prefix):
    
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


def make_data():
    train_temp_x,train_temp_y=load_data("train")
    test_temp_x,test_temp_y=load_data("test")
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    for j in range(0,10):
        count=0
        for i in range (0,len(train_temp_y)):
                if count==1000:
                    break
                if train_temp_y[i]==j:
                    train_x.append(train_temp_x[i])
                    train_y.append(train_temp_y[i])
                    count+=1


    for j in range(0,10):
        count=0
        for i in range (0,len(test_temp_y)):
                if count==100:
                    break
                if test_temp_y[i]==j:
                    test_x.append(test_temp_x[i])
                    test_y.append(test_temp_y[i])
                    count+=1

    return train_x,train_y,test_x,test_y

def hog_create(train_x,test_x):
    hog_train_x = [ hog(image) for image in train_x]
    hog_test_x= [ hog(image) for image in test_x]

    return hog_train_x,hog_test_x


def DTclassifier(hog_train_x,train_y,hog_test_x,test_y):



    print 'Accuracy for decision tree classifier on test data: '
   

    clf = DecisionTreeClassifier()
    clf.fit(hog_train_x,train_y)

    print clf.score(hog_test_x,test_y)

def RFclassifier(n,hog_train_x,train_y,hog_test_x,test_y):
    clf = RandomForestClassifier(n_estimators=n, n_jobs=-1)
    clf.fit(hog_train_x,train_y)

    print 'Accuracy for Random Forest classifier on test set with',n,'trees:'

    print clf.score(hog_test_x,test_y)


part=raw_input()

if part=='a':
    train_x,train_y,test_x,test_y=make_data()
    hog_train_x,hog_test_x=hog_create(train_x,test_x)
    DTclassifier(hog_train_x,train_y,hog_test_x,test_y)
if part=='b':
    train_x,train_y,test_x,test_y=make_data()
    hog_train_x,hog_test_x=hog_create(train_x,test_x)
    for n in range(5,10):
        RFclassifier(n,hog_train_x,train_y,hog_test_x,test_y)


