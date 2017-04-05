#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print enron_data["SKILLING JEFFREY K"].keys()
count =0
for pois in enron_data:
    if enron_data[pois]['poi'] == 1:
        count += 1
        
count = 0 




def open_file():    
    with open("C:/Users/O'l Mate/Documents/ML/Projects/ud120-projects/final_project/poi_names.txt") as names:
        x = names.readlines()
        for lines in x:
            if lines.startswith('('):
                count += 1
        print len(x)
        print count


for guys in enron_data:
    if enron_data[guys]['poi'] == 1:
        count += 1
        

print (count/len(enron_data))
leno = float(len(enron_data))
print leno
print float(float(count)/leno)




print count
