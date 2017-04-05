#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL')
data_dict.pop('LAVORATO JOHN J')
data_dict.pop('LAY KENNETH L')
data_dict.pop('FREVERT MARK A')
data_dict.pop('SKILLING JEFFREY K')

data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
### your code below

maxo = []

for shit, stack in data_dict.items():
    if stack['bonus'] != 'NaN' and stack['salary'] != 'NaN':
        if stack['bonus'] >= 5000000 or stack['salary'] >= 1000000:
            maxo.append([stack['bonus'], stack['salary'], shit])
print maxo


