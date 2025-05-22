# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary Python libraries for data handling, numerical computations, visualization, and machine learning.
2. Read the CSV file (student_scores.csv) into a DataFrame.
3. View the first few and last few rows of the dataset to understand its structure.
4. Separate the dataset into independent (x) and dependent (y) variables:
   
      x → input feature(s) (Hours studied).
   
      y → target/output (Scores obtained).
   
5. Split the dataset into training and testing sets

      test_size = 1/3 → 33% of data will be used for testing.
   
      random_state = 0 → ensures reproducibility.
   
6. Create and train a Linear Regression model on the training data.
7. Use the trained model to predict scores for the test dataset.
8. Calculate performance metrics:
    
      Mean Squared Error (MSE).
   
      Mean Absolute Error (MAE).
   
      Root Mean Squared Error (RMSE).
   
9. Plot the Training set:
    
      Scatter plot of actual points.
    
      Line plot of predicted regression line.
    
10. Plot the Testing set:
    
      Scatter plot of actual points.
       
      Line plot of predicted regression line based on test data.
## Program:

```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: D.Varshini
RegisterNumber: 212223230234

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datfile
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)

plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/4d1bdc3a-b13f-4a4f-9a4d-c0a93fce3d55)

![image](https://github.com/user-attachments/assets/14175561-2d69-4906-b4d9-2219c2322e04)

![image](https://github.com/user-attachments/assets/74f88050-4327-4033-964c-3e47284c19bd)

![image](https://github.com/user-attachments/assets/a8483f06-eda4-43e6-b64a-430162e46fee)

![image](https://github.com/user-attachments/assets/43cf1218-0e2c-4937-880a-f3fa2877a216)


![image](https://github.com/user-attachments/assets/7bfb3fec-b7ab-4d2f-b1c0-456f44642aa5)

![image](https://github.com/user-attachments/assets/14f4545b-ccbb-4c82-97a9-b5b2756816e6)

![image](https://github.com/user-attachments/assets/79659a03-5e1d-4429-bdfa-cf43173eaedc)

![image](https://github.com/user-attachments/assets/299c89fe-ee44-4400-8d32-2df1392cae62)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
