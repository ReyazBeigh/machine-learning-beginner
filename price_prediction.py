import pandas as pd

from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('input/melb_data.csv')

stats  = data.describe()

print(stats)

columns = data.columns

y = data.Price #prediction target

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = data[features]

print(X.describe()) #describe the data


model = DecisionTreeRegressor(random_state=1) #Define model 

fit = model.fit(X, y) #fit the model

print(fit) #print the model

print("Making predictions for the following 5 houses:")
print(X.head()) #print the first 5 rows
print("The predictions are")

print(model.predict(X.head())) #print the predictions for the first 5 rows