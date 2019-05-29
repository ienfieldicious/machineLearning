
#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset= pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x= LabelEncoder()

x[:,3]= labelencoder_x.fit_transform(x[:,3])

onehotencoder_x= OneHotEncoder(categorical_features=[3])
x=onehotencoder_x.fit_transform(x).toarray()

#avoiding dummy variables by not considering one categorical data

x=x[:,1:]

#splitting dataset into training and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)

#fitting multiple linearregression to training set

from sklearn.linear_model import LinearRegression

regressor= LinearRegression()

regressor.fit(x_train,y_train)


# predicting and not plotting

y_pred=regressor.predict(x_test)!

#building the optimal model using backward elimination

import statsmodels.formula.api as sm

x=np.append(arr = np.ones((50,1)).astype(int), values=x, axis=1)

x_opt=x[:,[0,1,2,3,4,5]]

regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()

regressor_ols.summary()

#removed variable with highest P value
x_opt=x[:,[0,1,3,4,5]]

regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()

regressor_ols.summary()

#removed variable with highest P value
x_opt=x[:,[0,3,4,5]]

regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()

regressor_ols.summary()

#removed variable with highest P value
x_opt=x[:,[0,3,5]]

regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()

regressor_ols.summary()


#removed variable with highest P value
x_opt=x[:,[0,3]]

regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()

regressor_ols.summary()

x=x[:,[0,3]]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)


opt_regressor= LinearRegression()

opt_regressor.fit(x_train,y_train)


# predicting and not plotting

y_pred2=opt_regressor.predict(x_test)