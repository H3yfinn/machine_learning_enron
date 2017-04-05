#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()





#########################################################
### your code goes here ###
clf = svm.SVC(kernel='linear')

t1= time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t1, 3), "s"

t2 = time()
pred = clf.predict(features_test)
print "training time:", round(time()-t2, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print accuracy
#########################################################
print 'break'

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

### your code goes here ###
clf = svm.SVC(kernel='linear')

t1= time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t1, 3), "s"

t2 = time()
pred = clf.predict(features_test)
print "training time:", round(time()-t2, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print accuracy

print 'break'

clf = svm.SVC(kernel='rbf')

t1= time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t1, 3), "s"

t2 = time()
pred = clf.predict(features_test)
print "training time:", round(time()-t2, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print accuracy

print 'break' 

clf = svm.SVC(kernel='rbf', C=10)

t1= time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t1, 3), "s"

t2 = time()
pred = clf.predict(features_test)
print "training time:", round(time()-t2, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print accuracy