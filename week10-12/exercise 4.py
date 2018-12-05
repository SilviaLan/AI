import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data"

data = pd.read_csv('LasVegasTripAdvisorReviews-Dataset.csv',sep = ';')

print(data.head())
attributes = data[['Native English','Instructer','Course','Semester','class size']]
target = data[['class attribute']]

#labels
plt.hist(data['class attribute'])
plt.xlabel('class attribute')
plt.ylabel('amount')
plt.show()

x = data['Course']
y = data['class attribute']

plt.scatter(x,y)
plt.xlabel('Course')
plt.ylabel('class attribute')
plt.show()

attributes_training = attributes[attributes.index % 2 != 0 ]
target_training = target[target.index %2 != 0 ]
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(attributes_training,target_training)

attributes_test = attributes[attributes.index %2 != 1 ]

# print attributes_test
actual_test = target[target.index %2 != 1]
actual_test.index = range(76)

prediction = clf.predict(attributes_test)
prediction_df = pd.DataFrame({'prediction': prediction})

# print prediction_data
# print actual_test
training_result = pd.concat([prediction_df,actual_test],axis=1)

# print training_result
misclassification = 0
for i in range(0,len(training_result)):
    if training_result['class attribute'][i]!=training_result['prediction'][i]:
        misclassification+=1

percent = float(misclassification)/len(training_result)*100
print ("Percent of misclassification = {:f} %".format(percent))