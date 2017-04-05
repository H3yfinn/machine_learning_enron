#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()

features_train = features_train[:len(features_train)/2]
labels_train = labels_train[:len(labels_train)/2]

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
def Adaboost():
    from sklearn import ensemble
    clf = ensemble.AdaBoostClassifier()
    
    t1=time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t1, 3), "s"
    
    t2 = time()
    pred = clf.predict(features_test)
    print "training time:", round(time()-t2, 3), "s"
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    print 'adaaccuracy', accuracy
    
def random_forest():
    from sklearn import ensemble
    clf = ensemble.RandomForestClassifier()
    
    t1=time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t1, 3), "s"
    
    t2 = time()
    pred = clf.predict(features_test)
    print "training time:", round(time()-t2, 3), "s"
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    print 'randomfor accuracy', accuracy
    return clf


def Knegihbours():
    try:
        from sklearn import neighbours
    except [ImportError, UnboundLocalError]:
        pass
    clf = neighbours.KNeighboursClassifier()
    
    t1=time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t1, 3), "s"
    
    t2 = time()
    pred = clf.predict(features_test)
    print "training time:", round(time()-t2, 3), "s"
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    print 'Kneigh accuracy', accuracy
    return clf

clf = Adaboost()    
clf = random_forest()
clf = Knegihbours()
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    print 'nameerror'

