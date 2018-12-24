#Put your answer here. (step 1 - copy the code from the previous question).

### max leaf nodes

# Import pandas to read in data
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
#%matplotlib inline
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score



# Read data using pandas
data = pd.read_csv("data/cell2cell.csv")

# Split into X and Y
X = data.drop(['churndep'], 1)
Y = data['churndep']

leaves = range(2,20)
accuracies = []

for leaf in leaves:
    tree = DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=leaf,)

    cross_fold_accuracies = cross_val_score(tree, X, Y, scoring="accuracy", cv=10)
    average_cross_fold_accuracy = np.mean(cross_fold_accuracies)
    accuracies.append(average_cross_fold_accuracy)

plt.plot(leaves, accuracies)
plt.xlabel("Max Leaf Nodes")
plt.ylabel("Accuracy")
plt.show()
