from os import listdir
from os.path import isfile, join
import sys
import numpy
import cPickle as pickle
import collections, re
import scipy.sparse
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2
import numpy as np
from sklearn import svm, ensemble , tree
from sklearn.metrics import confusion_matrix
import pylab as pl
from cPickle import dump, load
from skimage.feature import hog
import skimage
from matplotlib.colors import ListedColormap
from sklearn.utils import shuffle
import numpy as np
from sklearn import neighbors, datasets
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
from sklearn.decomposition import PCA

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

# help sought from https://github.com/jsantarc/Kernel-Nearest-Neighbor-Algorithm-in-Python-
def KernelKNNClassifierFit(X,Y,Kernel,Parameters):
    Y= numpy.array(Y)
    #Number of training samples
    N=len(X);
    #Array sorting value of kernels k(x,x)
    Gram_Matrix=numpy.zeros(N);
    #Calculate Kernel vector between same vectors Kxx[n]=k(X[n,:],X[n,:])
    #dummy for kernel name  
    for n in range(0,N):  
       
       Gram_Matrix[n]=pairwise_kernels(X[n], metric=Kernel,filter_params=Parameters)
    return Gram_Matrix

def predict(X_test,X_train,Kernel,Parameters, Gram_Matrix, Y_train, n_neighbors=1):
    Y_train=np.array(Y_train)
    Nz=len(X_test)
    #Empty list of predictions   
    yhat= numpy.zeros(Nz);
    #number of samples for classification
    #Number of training samples
    Nx=len(X_train);
    #Dummy variable  Vector of ones used to get rid of one loop for k(z,z) 
    Ones=numpy.ones(Nx);
    
    #squared Euclidean distance in kernel space for each training sample
    Distance=numpy.zeros(Nx)
    # Index of sorted values 
    Index= numpy.zeros(Nx)
    
    # calculate pairwise kernels beteen Training samples and prediction samples   
    Kxz=pairwise_kernels(X_train,X_test, metric=Kernel,filter_params=Parameters)
    
    NaborsNumberAdress=range(0,n_neighbors)

    #Calculate Kernel vector between same vectors Kxx[n]=k(Z[n,:],Z[n,:]) 
    
    for n in range(0,Nz):
        # calculate squared Euclidean distance in kernel space for each training sample 
        #for one prediction
        #for m in range(0,Nx)
        #Distance[m]=|phi(x[m])-phi(z[n])|^2=k(x,x)+k(z,z)-2k(z,x)    
                
        Distance =Gram_Matrix+pairwise_kernels(X_test[n], metric=Kernel,filter_params=Parameters)*Ones-2*Kxz[:,n]   
    
        #Distance indexes sorted from smallest to largest  
        Index=numpy.argsort(Distance.flatten());
        Index=Index.astype(int)

        #get the labels of the nearest feature vectors          
        yl=list(Y_train[Index[NaborsNumberAdress]])
        #perform majority vote 
        yhat[n]=max(set(yl), key=yl.count)
    return(yhat)

def crossValidation(data,Parameters):
    n_neighbors=4
    accuracy=0              # with frequency
    for i in [0]:
        testSet = [2*i+1,2*i+2]
        X_train, Y_train,X_test,Y_test = createCorpus(data,testSet, binaryX="False", stopWords="english", lemmatize="False")  # with frequency   
        X_train= X_train.todense()
        X_test=X_test.todense()
        # print "Fitting"
        num_samples_to_plot = 1000
        X_train, Y_train = shuffle(X_train, Y_train)
        X_train, Y_train = X_train[:num_samples_to_plot], Y_train[:num_samples_to_plot] 
        pca = PCA(n_components=2)
        X = pca.fit_transform(X_train)

        print X.shape
        y=Y_train


        h = .02  # step size in the mesh
        n_neighbors=4
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        for k in ['cosine', 'linear', 'rbf']:
            # we create an instance of Neighbours Classifier and fit the data.
            if k=='linear':
                clf = KernelKNNClassifierFit(X,y,'linear',0)
                clf1 = KernelKNNClassifierFit(X_train,Y_train,'linear',0)
                Z1 = predict(X_test, X_train,'linear',0,clf1,Y_train,n_neighbors)
                accuracy = np.mean(Z1==Y_test)
                print accuracy
            elif k=='cosine':
                clf= KernelKNNClassifierFit(X,y,'cosine',1)
                clf1 = KernelKNNClassifierFit(X_train,Y_train,'cosine',1)
                Z1 = predict(X_test, X_train,'cosine',1,clf1,Y_train,n_neighbors)
                accuracy = np.mean(Z1==Y_test)
                print accuracy
            elif k=='cosine':
                clf= KernelKNNClassifierFit(X,y,'rbf',1)
                clf1 = KernelKNNClassifierFit(X_train,Y_train,'rbf',1)
                Z1 = predict(X_test, X_train,'rbf',1,clf1,Y_train,n_neighbors)
                accuracy = np.mean(Z1==Y_test)
                print accuracy


            # I have gramMatrix in clf

            
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            if k=='linear':
                Z=predict(np.c_[xx.ravel(), yy.ravel()], X,'linear',0, clf,y,n_neighbors)
            elif k=='cosine':
                Z=predict(np.c_[xx.ravel(), yy.ravel()], X,'cosine',1,clf,y,n_neighbors)
            elif k=='rbf':
                Z=predict(np.c_[xx.ravel(), yy.ravel()], X,'rbf',1,clf,y,n_neighbors)
            print Z
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())

            plt.title("2-Class classification KernelKNN (kernel = "+k +")\nAccuracy="+str(accuracy))
            plt.savefig("KNN-SPAM"+k+".png")




if __name__ == "__main__":
    loadedData=pickle.load( open( "loadedData", "rb" ) )
    # questions = ["3a","3b","3c", "2ab"]
    # questions = ["1a"]
    #for q in questions:
    #   print "---------------------------"
    for C in [1]:
        crossValidation(loadedData,C)
    #   print "--------------------------"

#RBF: Accuracy  0.886620992173  Gamma 0.1
# Cosine Accuracy  0.888695925627  Gamma 1

