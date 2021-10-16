import pandas as pd
import plotly.express as px

df = pd.read_csv("logistics_data.csv")

salary = df["EstimatedSalary"].tolist()
purchased = df["Purchased"].tolist()

print(len(salary))

import plotly.graph_objects as go
salaries = df["EstimatedSalary"].tolist()

ages= df["Age"].tolist()
colors=[]
for data in purchased:
    if data == 1:
        colors.append("green")
    else:
        colors.append("red")
fig = go.Figure(data = go.Scatter(x = salaries, y = ages, mode = 'markers', marker = dict(color = colors)))

fig.show()
factors = df[["EstimatedSalary", "Age"]]
purchases = df["Purchased"]

from sklearn.model_selection import train_test_split
salary_train,salary_test, purchase_train, purchase_test = train_test_split(factors,purchases,test_size = 0.25, random_state = 0)

print(salary_train[0:10])

from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 

salary_train = sc_x.fit_transform(salary_train)  
salary_test = sc_x.transform(salary_test) 
  
print (salary_train[0:10])

from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(salary_train, purchase_train)
purchase_pred = classifier.predict(salary_test) 
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(purchase_test, purchase_pred))

user_age = int(input("Enter age of the customer -> "))
user_salary = int(input("Enter the salary of the customer -> "))

user_test = sc_x.transform([[user_salary, user_age]])

user_purchase_pred = classifier.predict(user_test)

if user_purchase_pred[0] == 1:
  print("This customer may purchase the product!")
else:
  print("This customer may not purchase the product!")