from array import array as py_array
from numpy import arange, array, int8, uint8, zeros
from os import chmod, remove, rmdir, walk
from os.path import join
from requests import head
from stat import S_IWUSR
from struct import unpack
from tempfile import mkdtemp
from itertools import product
from numpy import array, ceil, floor, nan, set_printoptions, sqrt, zeros
from sklearn.neighbors import KNeighborsClassifier

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

    indices = [i for i in range(nb_digits) if labels[i] in digits]
    nb_digits = len(indices)

    my_images = zeros((nb_digits, nb_rows*nb_cols), dtype=uint8)
    my_labels = zeros((nb_digits,), dtype=int8)
    
    for i in range(nb_digits):
        index = indices[i]
        my_images[i] = array(images[(index * nb_pixels):((index + 1) * nb_pixels)]) 
        my_labels[i] = labels[index]


    return my_images,my_labels

def classifier():
    train_images,train_labels = processData('train')
    test_images,test_labels   = processData('t10k')

    for k in [1,2,3,4]:
        for p in [1,2,3,4]:
            knn = KNeighborsClassifier(n_neighbors = k, p = p)
            knn.fit(train_images,train_labels) 
            print 'P=',p,'K=',k, ' Accuracy=', knn.score(test_images,test_labels)


classifier()
