import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



df = pd.read_csv('Titanic-Dataset.csv')


df = df.drop("Name", axis = 1, inplace=False)
# print(df.describe())
df = df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=False)

df['Age'] = df['Age'].fillna(df['Age'].mean())


df = df.drop("Embarked", axis = 1, inplace = False)
df['Sex'] = df['Sex'].replace("male", 0)
df['Sex'] = df['Sex'].replace("female", 1)
y = df['Survived']

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

X_scaled = scaler.fit_transform(X)

def calc(X, w, b):
    return np.dot(X, w.T) + b
def sigmoid(x):
    return 1/ (1 + np.exp(-x))


def logistic_regression(X, y, iterations):
    rows, columns = X.shape
    w = np.zeros(columns)
    b = 0
    for _ in range(iterations):
        z = calc(X, w, b)
        a = sigmoid(z)
        dz = a - y
        dw = np.dot(dz.T, X) / rows
        db = np.sum(dz) / rows


        w = w - 0.0001 * dw
        b = b - 0.0001 * db

    return w, b

w , b = logistic_regression(X_scaled, y, 1000)
def predict(Pclass, Sex, Age, sib, parch, fare, w, b):
    x = pd.DataFrame([[Pclass, Sex, Age, sib, parch, fare]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
    x_scaled = scaler.transform(x)
    z = calc(x_scaled, w, b)
    return sigmoid(z)


a = predict(3,1,22,1,0,7.25,w,b)
if(a>0.5):
    print(a)
    print("survived")
else:
    print(a)
    print("Died")









