Investigating Enron's scandal using Machine Learning

Introduction

In addition to being the largest bankruptcy reorganization in American history at that time, Enron was cited as the biggest audit failure
From a $90 price per share, to a $1 value represents the huge value loss and scam that happened in Enron. This case has been a point of interest for machine learning analysis because of the huge real-world impact that ML could help out and try to figure out what went wrong and how to avoid it in the future. It would be of great value to find a model that could potentially predict these types of events before much damage is done, so as to permit preventive action. Corporate governance, the stock market, and even the Government would be quite interested in a machine learning model that could signal potential fraud detections before hand.

Enron Data

The interesting and hard part of the dataset is that the distribution of the non-POI's to POI's is very skewed, given that from the 146 there are only 11 people or data points labeled as POI's or guilty of fraud. We are interested in labeling every person in the dataset into either a POI or a non-POI (POI stands for Person Of Interest). More than that, if we can assign a probability to each person to see what is the chance she is POI, it would be a much more reasonable model given that there is always some uncertainty.

Data Processing

All features in the dataset are either financial data or features extracted from emails. Financial data includes features like salary and bonus while the email features include number of messages written/received and to whom/form.

There are 2 clear outliers in the data, TOTAL and THE TRAVEL AGENCY IN THE PARK. The first one seems to be the sum total of all the other data points, while the second outlier is quite bizarre. Both these outliers are removed from the dataset for all the analysis. Also all features are scaled using the MinMaxScaler (although it is not included in the final model).

New features
3 new features were added:

poi_interaction:	POI related messages divided over the total messages from the person
from_poi_prop: POI related messages to the person divided over the total messages to the person
to_poi_prop: POI related messages from the person divided over the total messages from the person

We can expect that the financial gains for POI's is actually non-linear -likewise for emails -, that is why I applied a logarithmic transformation to all the features in the dataset.

PCA
By tuning the parameters, we get the best classification results. From the 20 features in total, they are reduced to 10 principal components.

Algorithms selection and tuning

For the analysis of the data, a total of 6 classifiers were tried out, which include:

Logistic Regression
Decision Tree Classifier
Gaussian Naive Bayes
Support Vector Classifier (LinearSVC)
AdaBoost
Random Forrest Tree Classifier


The object of the algorithm is to classify and find out which people are more likely to be POI's. There are clearly 2 categories we are looking to label the data.

To tune the overall performance, both automated and manual tuning of parameters was involved. The automated tuned parameters where done using the GridSearchCV from SkLearn.

Validation and Performance

To validate the performance of each algorithm, recall, precision and F1 scores where calculated for each one. You can find below a summary of the scores of the top algorithms.

    Feature	      F1 Score	Recall	Precision
Logistic Regression 0.77    0.73    0.85       
Results WILL vary. There is some randomness in the data splitting			
The best classifier was actually a Logistic Regression using PCA beforehand. This was achieved by using sklearn Pipline. The logistic regression achieved a consistent score above 0.30 for both precision and recall. The final parameters that were used are detailed below:

Paramaters: {'SEL__n_components': 10, 'CLF__C': 32, 'CLF__class_weight': 'balanced', 'CLF__random_state': 40}
It seems that the most important parameter to tune was to set the class_weight to balanced. I suspect this is due to the skewed nature of the dataset, because class weight assigns the importance of each class (POI or non-POI) depending on the inverse appearance of the class. So it set a much higher importance to POI's class which is exactly what we want in this case.

Discussion and Conclusions

This was just a starting point analysis for classifying Enron employees. The results should not be taken too seriously and more advanced models should be used. Possibilities for future research could be to include more complex pipelines for the data, or even Neural Networks. Here we tried a basic neural network, but the SkLearn library is very limited in what it has to offer in this regard.

How to run the code

To run the code, make sure you are located in the folder final_project. Then just run the following command:

python final_project.py
