import load_data_1

from sklearn.tree import DecisionTreeRegressor

y = load_data_1.melb_data.Price

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = load_data_1.melb_data[features]

mlb_model = DecisionTreeRegressor(random_state=0)

mlb_model.fit(X, y)

print("Prediction for only first five houses from the data that has already been used for training")

predicted = mlb_model.predict(X.head())

print(predicted)

print("Origional data")

print(y.head())