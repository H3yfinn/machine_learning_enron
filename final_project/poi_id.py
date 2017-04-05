#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = features_list = [
                'poi',
                'salary',
                'poi_interaction',
                 'to_messages',
                 'deferral_payments',
                 'total_payments',
                 'bonus',
                 'from_poi_prop',
                 'total_stock_value',
                 'shared_receipt_with_poi',
                 'from_poi_to_this_person',
                 'exercised_stock_options',
                 'from_messages',
                 'other',
                 'from_this_person_to_poi',
                 'to_poi_prop',
                 'deferred_income',
                 'expenses',
                 'restricted_stock',
                 'long_term_incentive'
                   ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#move data into dataframe
import pandas as pd
import numpy as np
data_dframe = pd.DataFrame.from_dict(data_dict, orient='index')
data_dframe = data_dframe.replace('NaN', np.nan)
df = data_dframe
### Task 2: Remove outliers
df.ix['BELFER ROBERT','total_payments'] = 3285
df.ix['BELFER ROBERT','deferral_payments'] = 0
df.ix['BELFER ROBERT','restricted_stock'] = 44093
df.ix['BELFER ROBERT','restricted_stock_deferred'] = -44093
df.ix['BELFER ROBERT','total_stock_value'] = 0
df.ix['BELFER ROBERT','director_fees'] = 102500
df.ix['BELFER ROBERT','deferred_income'] = -102500
df.ix['BELFER ROBERT','exercised_stock_options'] = 0
df.ix['BELFER ROBERT','expenses'] = 3285
df.ix['BELFER ROBERT',]
df.ix['BHATNAGAR SANJAY','expenses'] = 137864
df.ix['BHATNAGAR SANJAY','total_payments'] = 137864
df.ix['BHATNAGAR SANJAY','exercised_stock_options'] = 1.54563e+07
df.ix['BHATNAGAR SANJAY','restricted_stock'] = 2.60449e+06
df.ix['BHATNAGAR SANJAY','restricted_stock_deferred'] = -2.60449e+06
df.ix['BHATNAGAR SANJAY','other'] = 0
df.ix['BHATNAGAR SANJAY','director_fees'] = 0
df.ix['BHATNAGAR SANJAY','total_stock_value'] = 1.54563e+07
df.ix['BHATNAGAR SANJAY']

df = df.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'])
df = df.drop(['loan_advances', 'restricted_stock_deferred', 'director_fees', 'email_address'] , 1)    
### Task 3: Create new feature(s)
df['from_poi_prop'] = df['from_this_person_to_poi'] / df['to_messages']
df['to_poi_prop'] = df['from_poi_to_this_person'] / df['from_messages']
df['poi_interaction'] = (df['from_this_person_to_poi'] + df['from_poi_to_this_person']) / (df['to_messages'] + df['from_messages'])


#create new log scaled dataframe:
features_to_test = ['salary', 'poi_interaction', 'to_messages', 'deferral_payments', 'total_payments', 'bonus', 'from_poi_prop', 'total_stock_value', 'shared_receipt_with_poi', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'to_poi_prop', 'deferred_income', 'expenses', 'restricted_stock', 'long_term_incentive']

log_df = df
for f in features_to_test:
    log_df[f] = np.log10(df[f] + 1)
    #added +1 so the 0 values remained natural zeros
features_to_test.insert(0, 'poi')


log_df = log_df[['poi', 'salary', 'poi_interaction', 'to_messages', 'deferral_payments', 'total_payments', 'bonus', 'from_poi_prop', 'total_stock_value', 'shared_receipt_with_poi', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'to_poi_prop', 'deferred_income', 'expenses', 'restricted_stock', 'long_term_incentive']]

#have to get rid of NaN values before i do any minmax scaling (surprise!) Hopefully this isn't going to harm the results. 
from sklearn import preprocessing

nan_remover = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
log_df = nan_remover.fit_transform(log_df)

### Store to my_dataset for easy export below.
my_dataset = log_df

### Extract features and labels from dataset for local testing
labels, features = targetFeatureSplit(log_df)

#scale features using MinMaxscaler!
mmscaler = preprocessing.MinMaxScaler()
features = mmscaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#MACHINE LEARNING TIME!!
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from time import time
sep = '##############################################################################################'
sep2 = '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

##############################################################################################

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=40)
target_names = ['NonPoi', 'Poi']

##############################################################################################
#do logistic regression on data using paramters found using gridsearchCV
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



t = time()
pca = PCA(n_components=10)
logreg = linear_model.LogisticRegression(C=32, class_weight='balanced', random_state=40)
pipe = Pipeline(steps=[('PCA', pca),('LOG', logreg)])

pipe.fit(features_train, labels_train)
pred = pipe.predict(features_test)
print "training time:", round(time()-t, 3), "s"
print classification_report(labels_test, pred, target_names=target_names)
print 'accuracy=', accuracy_score(labels_test, pred)




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
features_list = [
                'poi',
                'salary',
                'poi_interaction',
                 'to_messages',
                 'deferral_payments',
                 'total_payments',
                 'bonus',
                 'from_poi_prop',
                 'total_stock_value',
                 'shared_receipt_with_poi',
                 'from_poi_to_this_person',
                 'exercised_stock_options',
                 'from_messages',
                 'other',
                 'from_this_person_to_poi',
                 'to_poi_prop',
                 'deferred_income',
                 'expenses',
                 'restricted_stock',
                 'long_term_incentive'
                   ]

dump_classifier_and_data(pipe, log_df, features_list)

