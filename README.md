# Creating-Wine-Review-System
##Introduction and Data
For this lab, we will utilize the “Wine Quality” dataset from the UCI Machine Learning Repository.

UCI MLR: Wine Quality
While the data could be obtained from this website, we will provide a specific and slightly pre-processed version of the data. However, this is a good website to be aware of! It houses many datasets that are useful for practice training machine learning models.

To complete this lab, you’ll need to import the following:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
wine_white = pd.read_csv("https://cs307.org/lab/lab-04/data/winequality-white.csv", delimiter=";")
```
the columns that represent the features are:

fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
The target is quality. It’s meaning, for the original paper, is:

Regarding the preferences, each sample was evaluated by a minimum of three sensory assessors (using blind tastes), which graded the wine in a scale that ranges from 0 (very bad) to 10 (excellent).

In this case, the features are the physicochemical data. That is, these are chemical properties of the individual wines that can be measured in a lab. For our purposes, we are not super interested in the details of these, but if you are interested, additional details are given in the original paper. But for the purposes of this lab, assume that there is some fixed cost to process and obtain these 11 measurements for any wine, but that doing so is far cheaper than paying humans to taste and review the wine.

The target is the sensory data, as “measured” by humans tasting the wine and reviewing it.

*Goal: Find a model that is useful for predicting the quality of a wine based on its physicochemical properties, for the purpose of potentially removing the need for human testers.

Before we begin modeling, let’s take care of some pre-processing, in particular:

Specifying the features and target
Moving from pandas to numpy
Scale the X data
Note: We are simply going to do this before the test-train split. This is not necessarily recommended in practice, but instead is for ease of completing the lab. Because we are scaling all the data immediately, you do not have to worry about scaling at all. Not at train time, and not at test time.
Splitting the data for training, validation, and testing
```python
# specify target and feature variables
target = ['quality']
features = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

# create numpy arrays
X = wine_white[features].to_numpy()
y = wine_white[target].to_numpy().ravel()

# scale the data, warning: do not do it like this in practice (more on this later)
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)

# create train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# create validation-train and validation datasets
X_vtrain, X_val, y_vtrain, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42
)
```
As a reminder:

_train indicates a full train dataset
_vtrain is a validation-train dataset that we will use to fit models
_val is a validation dataset that we will use to select models, in this case, select a value of k, based on models that were fit to a validation-train dataset.
_test is a test dataset that we will use to report an estimate of generalization error, after first refitting a chosen model to a full train dataset

##Model Training 

```python
# use this cell for the linear model
linear=LinearRegression()
linear.fit(X_vtrain, y_vtrain)
rmseLin= np.sqrt(np.mean((linear.predict(X_val)-y_val)**2))

#hyper parameter
lasso_val_rmse=[]
for a in alpha:
    lasso = Lasso(alpha=a)
    lasso.fit(X_vtrain, y_vtrain)
    pred = lasso.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    lasso_val_rmse.append(rmse)
print(lasso_val_rmse)
print(alpha[lasso_val_rmse.index(min(lasso_val_rmse))],"RMSE:", min(lasso_val_rmse))

lasso = Lasso(alpha=alpha[2]) #hyper parameter
lasso.fit(X_train,y_train) #train
pred = lasso.predict(X_test) #test
rmse = np.sqrt(mean_squared_error(y_test,pred))
rmse
# use this cell for the ridge model
#determining the hyper parameter
ridge_val_rmse = []
for a in alpha:
    ridge = Ridge(alpha=a)
    ridge.fit(X_vtrain,y_vtrain)
    pred = ridge.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val,pred))
    ridge_val_rmse.append(rmse)


print(ridge_val_rmse)
print(alpha[ridge_val_rmse.index(min(ridge_val_rmse))],"RMSE:", min(ridge_val_rmse))

#trained final model
ridge = Ridge(alpha=alpha[4]) #hyper parameter
ridge.fit(X_train,y_train) #train
pred = ridge.predict(X_test) #test
rmse = np.sqrt(mean_squared_error(y_test,pred))
rmse

dt = DecisionTreeRegressor(max_depth=5) # define model based on current k
dt.fit(X_train, y_train) # fit model to the validation-train data
pred = dt.predict(X_test) # make predictions with validation data
dtrmse = np.sqrt(mean_squared_error(y_test, pred)) # calculate validation RMSE
dtrmse

##Graph
```python
plt.suptitle('Predicted versus Actual')

# create subplot for Champaign
plt.title("Random Forest")
plt.scatter(y_test, pred_rf, color="dodgerblue")
plt.xlabel("Actual")
plt.ylabel("Prediction")
plt.grid(True, linestyle='--', color='lightgrey')
plt.axline((0,0),slope = 1, color ='black')
```

<img width="561" alt="predvsact" src="https://github.com/user-attachments/assets/6f17dec5-e686-430c-9cfa-ac8c9d2e0e17" />
