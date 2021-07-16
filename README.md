# Score prediction
 Predicting the percentage of an student based on the no. of study hours using supervised ML. This is a simple linear regression using python.
 
 # Student score prediction using supervised ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv('student_scores.csv')
df

# Data Plot

df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('score(In percentage)')  
plt.show()

# labeling the data

X = df.iloc[:, :-1].values 
X =X.reshape(-1,1)
y = df.iloc[:, 1].values
y = y.reshape (-1,1)

# Train test split using sklearn built in function

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)

# Training algorithm

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training complete.")

# Plotting regression line

line = regressor.coef_*X+regressor.intercept_

plt.title('Plotting for the test data') 
plt.scatter(X, y)
plt.plot(X, line, color = 'y');
plt.show()


# Time for prediction

print("Testing Data")
print(X_test) 
y_pred = regressor.predict(X_test)

# Actual vs Predicted data

ev = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
ev

# Testing the prediction model

hours = np.array([9.25]) 
hours = hours.reshape(-1,1)
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(float(hours)))
print("Predicted Score = {}".format(round(own_pred[0],2)))

# Model evaluation

# Importing metrics from sklearn 
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

# To find Mean Absolute Error(mse)
mse = (mean_absolute_error(y_test, y_pred))
print("MAE:",mse)

# To find Root Mean Squared Error(rmse)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print("RMSE:",rmse)

# To find coefficient of determination
r2 =  r2_score(y_test, y_pred)
print("R-Square:",r2)
