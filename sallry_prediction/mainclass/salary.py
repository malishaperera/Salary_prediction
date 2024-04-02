import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#import data

dataset = pd.read_csv('../data/salary_data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting data
#test data original

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

#fiting data
#data 2/3

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting
y_pred = regressor.predict(X_test)


#Visulising the Training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')


plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()























