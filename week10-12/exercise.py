import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
response = urllib.request.urlopen(url)
data = response.read()      # a raw bits 'bytes' object
text = data.decode('utf-8') # use the utf-8 string format to create a string 'str' object
iris_df=pd.read_csv(url, names=("sepal length","sepal width","petal length","petal width","class")) # Panda object

iris_df[:].head()

#3.2.1 example
x = iris_df["sepal length"]
y = iris_df["sepal width"]
iris_df["class"]
colors = {'Iris-setosa':'red', 'Iris-virginica':'blue', 'Iris-versicolor':'green'}

plt.scatter(x, y, c = iris_df["class"].apply(lambda x: colors[x]))

#labels
plt.xlabel('sepal length')
plt.ylabel('sepal width')

#plt.show()

#3.2.1 question 1
x = iris_df["sepal length"]
y = iris_df["petal length"]
z = iris_df["petal width"]
classes = iris_df["class"]

fig = plt.figure()

ax = fig.add_subplot(111,projection = '3d')

colors = {'Iris-setosa':'red','Iris-virginica':'blue','Iris-versicolor':'green'}

for i in range(len(x)):
    ax.scatter(x[i],y[i],z[i], c = colors[classes[i]])

#labels
ax.set_xlabel('sepal length')
ax.set_ylabel('petal length')
ax.set_zlabel('petal width')

plt.show()
