x# Machine Learning

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Fetch File Data

DataSet = pd.read_csv('Salary_Data.csv')

x = DataSet.iloc[:,:-1].values

y= DataSet.iloc[:,1].values


# Missing Data management

"""from sklearn.impute import SimpleImputer

simpleimputer = SimpleImputer(missing_values=np.nan, strategy="mean", copy= True)

simpleimputer = simpleimputer.fit(x[:,1:3])

x[:,1:3]= simpleimputer.transform(x[:,1:3])"""



# Encoding Categorical Data

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x= LabelEncoder()

x[:,0]= labelencoder_x.fit_transform(x[:,0])

onehotencoder = OneHotEncoder(categorical_features=[0])

x= onehotencoder.fit_transform(x).toarray()

labelencoder_y= LabelEncoder()

y=labelencoder_y.fit_transform(y)"""


# SPlitting TRaining and testing set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3, random_state=0 )

# Feature Scaling

"""from sklearn.preprocessing import StandardScaler

sc_x= StandardScaler()

x_train= sc_x.fit_transform(x_train)

x_test= sc_x.transform(x_test)

sc_y= StandardScaler()

y_train= sc_y.fit_transform(y_train) """

#Fitting simple linear regression to training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

# Making prediction

y_pred = regressor.predict(x_test)

# Visualizing the Training Set results

plt.scatter(x_train,y_train, color='red')

plt.plot(x_train, regressor.predict(x_train), color='blue')

plt.title('Salary vs Experience (Training Set)')

plt.xlabel(" Years of experience")

plt.ylabel("Salary")

plt.show()


# Visualizing the Test Set results
plt.scatter(x_test,y_test, color='red')

plt.plot(x_train, regressor.predict(x_train), color='blue')

plt.title('Salary vs Experience (Test Set)')

plt.xlabel(" Years of experience")

plt.ylabel("Salary")

plt.show()








