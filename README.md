# Regression-Tree-with-Regression-Model

Note: this is an experimental model



## What does this model do? 
Decision tree is a classic machine learning model. However, have you ever wondered why we use "mean" as the prediciton result in leaf nodes?

![image](https://miro.medium.com/max/875/1*ZVq5QHRyCdx-HJcpSCspKQ.png)

Why can't we use regression model instead? Does regression model in leaf nodes reduce the error of prediction? Let's find out!
![image](https://miro.medium.com/max/875/1*-s-RH_g23FqJ3-mnLZcm3Q.png)

## How to use this package?
First, git clone it.
```
git colne https://github.com/AllenShiah/Regression-Tree-with-Regression-Model.git
```
Second, import the package.
```
import RegressionTree
```
Third, use it like sklearn.
```C++
# exapmple
model = DecisionTreeRegression(min_samples_split=3, max_depth=3, kernel='regression')
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test) 
```
Note: this package has two kernels, average and regression, first one is the classic regression tree and the latter is the model I reformed. 
