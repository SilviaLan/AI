# Import pandas to read in data
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
%matplotlib inline


# Read data using pandas
data = pd.read_csv("data/cell2cell.csv")

# Split into X and Y
X = data.drop(['churndep'], 1)
Y = data['churndep']

C_parameters = range(1,50)
accuracies = []

for c in C_parameters:
    tree = LinearSVC(C = c)

    cross_fold_accuracies = cross_val_score(tree, X, Y, scoring="accuracy", cv=10)
    average_cross_fold_accuracy = np.mean(cross_fold_accuracies)
    accuracies.append(average_cross_fold_accuracy)

plt.plot(C_parameters, accuracies)
plt.xlabel("C-coefficient")
plt.ylabel("Accuracy")
plt.show()