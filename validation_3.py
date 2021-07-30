from sklearn.metrics import mean_absolute_error
import define_model_2 as mlb_model_cals

prediction_on_all_data = mlb_model_cals.mlb_model.predict(mlb_model_cals.X)

mean_absolute_error = mean_absolute_error(mlb_model_cals.y, prediction_on_all_data)

print("Mean Absolute Error: " + str(mean_absolute_error))

print(__file__ + " DONE ")