from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()

iris_Data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['target'])
iris_Data['target'] = iris_Data['target'].map({0: "setosa", 1:"versicolor", 2:"virginica"})
iris_Data.to_csv('iris_classification.csv')
print(iris_Data)

X_Data = iris_Data.iloc[:, :-1]
Y_Data = iris_Data.iloc[:, [-1]]

