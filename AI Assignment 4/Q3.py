import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# % matplotlib inline

# Read in our csv data
data = pd.read_csv("data/cell2cell.csv")

# Put all features into X and the target variable into Y
X = data.drop('churndep',axis = 1)
Y = data['churndep']

# Prepare to do some training and testing
training_percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
tree_accuracies = []
logistic_accuracies = []
SVC_accuracies = []

# Loop through your training percentages, split your data with each percentage,
#  create both models, fit/train both models, predict with your models and
#  append each accuracy to the correct list
for training_percentage in training_percentages:
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1-training_percentage)

    # Create both models
    model1 = DecisionTreeClassifier()
    model2 = LogisticRegression()
    model3 = LinearSVC()
    # Fit both model
    model1.fit(X_train,Y_train)
    model2.fit(X_train,Y_train)
    model3.fit(X_train,Y_train)

    # Get predictions from both models
    Y_test_model1 = model1.predict(X_test)
    Y_test_model2 = model2.predict(X_test)
    Y_test_model3 = model3.predict(X_test)

    # Get the accuracy for the models' predictions
    tree_acc = accuracy_score(Y_test,Y_test_model1)
    logistic_acc = accuracy_score(Y_test,Y_test_model2)
    SVC_acc = accuracy_score(Y_test, Y_test_model3)

    # get the confusion matrix
    confusion_matrix(...)

    # Now that I have a tree and logistic accuracy, I should add them to my list of accuracies
    tree_accuracies.append(tree_acc)
    logistic_accuracies.append(logistic_acc)
    SVC_accuracies.append(SVC_acc)

# get the confusion matrix
confusion_matrix(...)

# print accuracy and confusion matrix in a nicely formatted table
print(tree_accuracies)
print(logistic_accuracies)
print(SVC_accuracies)
