from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import a4_split_test_data as splitted_data

forest_model = RandomForestRegressor(random_state=0)
forest_model.fit(splitted_data.train_X, splitted_data.train_y)
prediction = forest_model.predict(splitted_data.test_X)
mea = mean_absolute_error(splitted_data.test_y, prediction)
print( " MEA from Random Forest Model, should be better from other 2 approaches "+str(mea))