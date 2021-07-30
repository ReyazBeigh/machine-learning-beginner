import define_model_2 as model_def
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

train_X,test_X,train_y,test_y = train_test_split(model_def.X,model_def.y,random_state=0)

mlb_model = DecisionTreeRegressor(max_depth=10)

mlb_model.fit(train_X,train_y)

real_prediction = mlb_model.predict(test_X)

print("MEA known data to see how good is our model")


print(mean_absolute_error(test_y,real_prediction))

print(__file__ + " DONE ")