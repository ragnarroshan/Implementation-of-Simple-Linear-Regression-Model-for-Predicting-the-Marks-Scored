# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
5. for each data point calculate the difference between the actual and predicted marks
6. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7. Once the model parameters are optimized, use the final equation to predict marks for any new input data

## Program:
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: kirthick roshan j

RegisterNumber: 212223040097
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('data.csv')

#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
### Head Values
![image](https://github.com/user-attachments/assets/f8383e2e-1bcc-4137-b1cd-fd706bd438fb)


### Tail Values

![image](https://github.com/user-attachments/assets/d3b67821-e803-4413-9f0f-415f6b04a21f)

### X Values
![image](https://github.com/user-attachments/assets/a4a3710b-6d70-4286-a728-cc641ba21638)


### y Values


### Predicted Values

![image](https://github.com/user-attachments/assets/eee55077-1f9e-4a60-9471-d5e2a131a6e8)

### Actual Values
![image](https://github.com/user-attachments/assets/0b4bed37-2dea-4330-abcc-cfb278976f69)


### Training Set
![image](https://github.com/user-attachments/assets/d8a0d3b8-d698-4bbf-a7d1-505f963a0734)


### Testing Set
![image](https://github.com/user-attachments/assets/83dc9c01-c442-4aaf-ad30-245eb1905a07)


### MSE, MAE and RMSE
![image](https://github.com/user-attachments/assets/c438fe10-89be-4e8c-9eda-0e436fbf83ca)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
