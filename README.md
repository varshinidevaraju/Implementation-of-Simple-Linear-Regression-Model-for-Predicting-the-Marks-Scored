# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Store data in a structured format (e.g., CSV, DataFrame).
2. Use a Simple Linear Regression model to fit the training data.
3. Use the trained model to predict values for the test set.
4.Evaluate performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VARSHINI D
RegisterNumber: 212223230234

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datfile
df.head()
*/
```
## Output:

![image](https://github.com/user-attachments/assets/d804cf16-5228-4db9-93d1-3428f75692cf)

```
df.tail()
```
## Output:
![image](https://github.com/user-attachments/assets/14175561-2d69-4906-b4d9-2219c2322e04)

```
x=df.iloc[:,:-1].values
x
```
## Output:
![image](https://github.com/user-attachments/assets/74f88050-4327-4033-964c-3e47284c19bd)

```
y=df.iloc[:,1].values
y
```
## Output:
![image](https://github.com/user-attachments/assets/a8483f06-eda4-43e6-b64a-430162e46fee)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
```
```
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
```

```
y_pred
```
## Output:
![image](https://github.com/user-attachments/assets/43cf1218-0e2c-4937-880a-f3fa2877a216)

```
y_test
```
## Output:
![image](https://github.com/user-attachments/assets/7bfb3fec-b7ab-4d2f-b1c0-456f44642aa5)
```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)
```
## Output:
![image](https://github.com/user-attachments/assets/14f4545b-ccbb-4c82-97a9-b5b2756816e6)
```
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/79659a03-5e1d-4429-bdfa-cf43173eaedc)

```
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/299c89fe-ee44-4400-8d32-2df1392cae62)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
