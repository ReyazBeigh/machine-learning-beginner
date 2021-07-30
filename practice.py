from numpy import empty
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

def get_mea(max_leaf_nodes,train_X,value_X,train_y,value_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_X,train_y)
    pred_y = model.predict(value_X)
    mae = mean_absolute_error(value_y,pred_y)
    return (mae)

data = pd.read_csv('input/melb_data.csv')

stats  = data.describe()

#print(stats)


columns = data.columns

y = data.Price #prediction target

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = data[features]

#print(X.describe()) #describe the data

train_X, val_X, train_y,val_y =  train_test_split(X,y,random_state=0)    

leaves = [5,50,500,600,5000]
optimal_tree_size = leaves[0]
temp_mae = None

print("Chooing Correct Number of Leaf nodes ")
for leaf in leaves:
    mae = get_mea(leaf,train_X,val_X,train_y,val_y)
    print("For Leaf nodes ",leaf," MAE is ",mae)
    if temp_mae == None or temp_mae > mae:
        temp_mae = mae
        optimal_tree_size = leaf


model = DecisionTreeRegressor(max_leaf_nodes=optimal_tree_size,random_state=0) #Define model 

fit = model.fit(X, y) #fit the model

#print(fit) #print the model

print("Making predictions for all the houses:")

print("The predictions are")

predection = model.predict(val_X)
print(predection) #print the predictions for the first 5 rows

mean_absolute_error = mean_absolute_error(val_y, predection)

print("mean_absolute_error is ")
print(mean_absolute_error) #print the mean absolute error

print("Random Forests")
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X,train_y)
prediction_forest = forest_model.predict(val_X)

mean_absolute_error = mean_absolute_error(val_y, prediction_forest)

print("Predection is %d and Mean absolute Error is %d",prediction_forest,mean_absolute_error)