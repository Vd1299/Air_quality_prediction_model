# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:31:39 2020

@author: vishvesh
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('Data/Real-Data/real_combine.xlsx')

sns.heatmap(df.isnull(),yticklabels=False, cbar=True, cmap='viridis')

df = df.dropna()

X = df.iloc[:,1:7]
Y= df.iloc[:,0]

X.isnull()
Y.isnull()

sns.pairplot(df)

df.corr()

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')

corrmat.index

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,Y)

print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show()

sns.distplot(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

##linear regression
from sklearn import linear_model
from sklearn import metrics
lm = linear_model.LinearRegression()
lm.fit(X_train, Y_train)

lm.coef_

lm.intercept_

lm.score(X_train, Y_train)
lm.score(X_test, Y_test)

from sklearn.model_selection import cross_val_score
score=cross_val_score(lm,X,Y,cv=5)
score.mean()
##to get coefficient of features
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df

prediction=lm.predict(X_test)

sns.distplot(Y_test-prediction)

plt.scatter(Y_test, prediction)


print('MAE:', metrics.mean_absolute_error(prediction, Y_test))
print('MSE:',metrics.mean_squared_error(prediction, Y_test))

##lasso
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
##to get mean from linear regression model
lin_regressor = LinearRegression()
mse = cross_val_score(lin_regressor,X,Y,scoring='neg_mean_squared_error',cv=5)
mean_mse = np.mean(mse)
print(mean_mse)

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,Y)
##gives best value of alpha
print(lasso_regressor.best_params_)
##gives the best score
print(lasso_regressor.best_score_)

prediction=lasso_regressor.predict(X_test)

sns.distplot(Y_test-prediction)

plt.scatter(Y_test, prediction)
print('MAE:', metrics.mean_absolute_error(prediction, Y_test))
print('MSE:',metrics.mean_squared_error(prediction, Y_test))