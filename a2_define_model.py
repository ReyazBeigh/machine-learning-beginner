import a1_load_data

from sklearn.tree import DecisionTreeRegressor

y = a1_load_data.melb_data.Price

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = a1_load_data.melb_data[features]

mlb_model = DecisionTreeRegressor(random_state=0)

mlb_model.fit(X, y)

print("Prediction for only first five houses from the data that has already been used for training")

predicted = mlb_model.predict(X.head())

print(predicted)

print("Origional data")

print(y.head())


print(__file__ + " DONE ")