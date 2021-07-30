
import a4_split_test_data as splitted_data
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

#find the best value for max_leaf_nodes
possible_best_nodes = [5, 10, 20, 30, 50, 100, 200,500,1000,2000,5000]

optimal_node = possible_best_nodes[0]

temp_mea = None

for leaf in possible_best_nodes:
        mae = get_mae(leaf, splitted_data.train_X, splitted_data.test_X, splitted_data.train_y, splitted_data.test_y)
        if temp_mea == None or temp_mea > mae:
            temp_mea = mae
            optimal_node = leaf

optimal_model = DecisionTreeRegressor(max_leaf_nodes=optimal_node, random_state=0)
optimal_model.fit(splitted_data.train_X, splitted_data.train_y)
preds_val = optimal_model.predict(splitted_data.test_X)
mae = mean_absolute_error(splitted_data.test_y, preds_val)

print("MEA after OPTIMISATION [in the middle of underfitting and overfitting]-> " + str(mae))

print(__file__+" DONE")