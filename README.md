# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Karan A
RegisterNumber:  212223230099
```
```py
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
confusion=confusion_matrix(y_test,y_pred)
confusion
```

## Output:
### DATA SAMPLE:
![image](https://github.com/user-attachments/assets/4d69e70f-6a49-4e19-9057-e264898f5f79)

### "LEFT" FEATURE VALUE COUNTS:
![image](https://github.com/user-attachments/assets/3858be02-fe47-42ea-8fb7-7c51df5c1315)
### ENCODED "SALARY":
![image](https://github.com/user-attachments/assets/ae96e849-0a12-48e6-a4aa-a792313f05fb)
### X:
![image](https://github.com/user-attachments/assets/168139b8-4770-4abb-b099-1d718448b273)
### Y:
![image](https://github.com/user-attachments/assets/9152779e-9986-4bf9-bf77-ae55dc1cf525)
### ACCURACY SCORE:
![image](https://github.com/user-attachments/assets/0a0aad23-a2fc-4057-9462-1fcb78a08265)
### CONFUSION MATRIX:
![image](https://github.com/user-attachments/assets/f04ee8d1-b2f0-48aa-acec-728c795c9183)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
