import numpy as np
import pandas as pd

# Import models and evaluation functions
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
#from sklearn import cross_validation

# Import vectorizers to turn text into numeric
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Import plotting
import matplotlib.pylab as plt
#%matplotlib inline

import scipy.sparse as sps
import matplotlib.pyplot as plt

data = pd.read_csv("data/books.csv", quotechar="\"", escapechar="\\")

X_text = data['review_text']
Y = data['positive']

# Create a vectorizer that will track text as binary features
binary_vectorizer = CountVectorizer(binary=True)

# Let the vectorizer learn what tokens exist in the text data
binary_vectorizer.fit(X_text)

# Turn these tokens into a numeric matrix
X = binary_vectorizer.transform(X_text)
#print(X)


features = binary_vectorizer.get_feature_names()
#features[10000:10020]

#plt.figure(figsize=(20,10))
#plt.spy(X.toarray())
#plt.show()
'''
def getWords(bag_of_words, file_index_row, features_list):
    ans = []
    for i in range(len(features_list)):
        if bag_of_words[file_index_row,i] ==1:
            ans.append(features_list[i])
    return ans
    
print(getWords(X,1,features))    
'''

# Create a model
logistic_regression = LogisticRegression()

# Use this model and our data to get 5-fold cross validation accuracy
acc = cross_validation.cross_val_score(logistic_regression, X, Y, scoring="accuracy", cv=5)

# Print out the average accuracy rounded to three decimal points
print ("Mean accuracy of our classifier is " + str(round(np.mean(acc), 3)) )

new_review = """"
really bad book!
"""

add_review = pd.DataFrame([[new_review,0]],columns=['review_text','positive'])

data_new = data.append(add_review)
X_text_new = data_new['review_text']
Y_new = data_new['positive']

# Create a vectorizer that will track text as binary features
binary_vectorizer = CountVectorizer(binary=True)

# Let the vectorizer learn what tokens exist in the text data
binary_vectorizer.fit(X_text_new)

# Turn these tokens into a numeric matrix
X_new = binary_vectorizer.transform(X_text_new)

# Create a model
logistic_regression = LogisticRegression()

# Use this model and our data to get 5-fold cross validation accuracy
acc = cross_validation.cross_val_score(logistic_regression, X_new, Y_new, scoring="accuracy", cv=5)

# Print out the average accuracy rounded to three decimal points
print ("Mean accuracy of the classifier with the new review is " + str(round(np.mean(acc), 3)) )







