from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import cProfile

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                    boston.target,
                                                    test_size=0.25,
                                                    random_state=0)

# Fit decision tree regression model
dt_mse_regressor = DecisionTreeRegressor(random_state=0)
dt_mae_regressor = DecisionTreeRegressor(random_state=0, criterion="mae")

print "Profiling MSE DecisionTreeRegressor"
pr = cProfile.Profile()
pr.enable()
dt_mse_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

print "Profiling MAE DecisionTreeRegressor"
pr = cProfile.Profile()
pr.enable()
dt_mae_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

dt_mse_predicted = dt_mse_regressor.predict(X_test)
dt_mae_predicted = dt_mae_regressor.predict(X_test)

dt_mse_mse = mean_squared_error(y_test, dt_mse_predicted)
dt_mae_mse = mean_squared_error(y_test, dt_mae_predicted)
print("Mean Squared Error of DecisionTreeRegressor "
      "Trained w/ MSE Criterion: {}").format(dt_mse_mse)
print("Mean Squared Error of DecisionTreeRegressor "
      "Trained w/ MAE Criterion: {}").format(dt_mae_mse)

dt_mse_mae = mean_absolute_error(y_test, dt_mse_predicted)
dt_mae_mae = mean_absolute_error(y_test, dt_mae_predicted)
print("Mean Absolute Error of DecisionTreeRegressor "
      "Trained w/ MSE Criterion: {}").format(dt_mse_mae)
print("Mean Absolute Error of DecisionTreeRegressor "
      "Trained w/ MAE Criterion: {}").format(dt_mae_mae)
print("")

# Fit Randomforestregressor regression model####################################
rf_mse_regressor = RandomForestRegressor(random_state=0)
rf_mae_regressor = RandomForestRegressor(random_state=0, criterion="mae")

print "Profiling MSE RandomForestRegressor"
pr = cProfile.Profile()
pr.enable()
rf_mse_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

print "Profiling MAE RandomForestRegressor"
pr = cProfile.Profile()
pr.enable()
rf_mae_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

rf_mse_predicted = rf_mse_regressor.predict(X_test)
rf_mae_predicted = rf_mae_regressor.predict(X_test)

rf_mse_mse = mean_squared_error(y_test, rf_mse_predicted)
rf_mae_mse = mean_squared_error(y_test, rf_mae_predicted)
print("Mean Squared Error of RandomForestRegressor "
      "Trained w/ MSE Criterion: {}").format(rf_mse_mse)
print("Mean Squared Error of RandomForestRegressor "
      "Trained w/ MAE Criterion: {}").format(rf_mae_mse)

rf_mse_mae = mean_absolute_error(y_test, rf_mse_predicted)
rf_mae_mae = mean_absolute_error(y_test, rf_mae_predicted)
print("Mean Absolute Error of RandomForestRegressor "
      "Trained w/ MSE Criterion: {}").format(rf_mse_mae)
print("Mean Absolute Error of RandomForestRegressor "
      "Trained w/ MAE Criterion: {}").format(rf_mae_mae)
print("")

# Fit GradientBoostingRegressor regression model####################################
gb_mse_regressor = GradientBoostingRegressor(random_state=0)
gb_mae_regressor = GradientBoostingRegressor(random_state=0, criterion="mae")

print "Profiling Friedman MSE GradientBoostingRegressor"
pr = cProfile.Profile()
pr.enable()
gb_mse_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

print "Profiling MAE GradientBoostingRegressor"
pr = cProfile.Profile()
pr.enable()
gb_mae_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

gb_mse_predicted = gb_mse_regressor.predict(X_test)
gb_mae_predicted = gb_mae_regressor.predict(X_test)

gb_mse_mse = mean_squared_error(y_test, gb_mse_predicted)
gb_mae_mse = mean_squared_error(y_test, gb_mae_predicted)
print("Mean Squared Error of GradientBoostingRegressor "
      "Trained w/ Friedman MSE Criterion: {}").format(gb_mse_mse)
print("Mean Squared Error of GradientBoostingRegressor "
      "Trained w/ MAE Criterion: {}").format(gb_mae_mse)

gb_mse_mae = mean_absolute_error(y_test, gb_mse_predicted)
gb_mae_mae = mean_absolute_error(y_test, gb_mae_predicted)
print("Mean Absolute Error of GradientBoostingRegressor "
      "Trained w/ Friedman MSE Criterion: {}").format(gb_mse_mae)
print("Mean Absolute Error of GradientBoostingRegressor "
      "Trained w/ MAE Criterion: {}").format(gb_mae_mae)
print("")

# Fit ExtraTreesRegressor regression model####################################
et_mse_regressor = ExtraTreesRegressor(random_state=0)
et_mae_regressor = ExtraTreesRegressor(random_state=0, criterion="mae")

print "Profiling MSE ExtraTreesRegressor"
pr = cProfile.Profile()
pr.enable()
et_mse_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

print "Profiling MAE ExtraTreesRegressor"
pr = cProfile.Profile()
pr.enable()
et_mae_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

et_mse_predicted = et_mse_regressor.predict(X_test)
et_mae_predicted = et_mae_regressor.predict(X_test)

et_mse_mse = mean_squared_error(y_test, et_mse_predicted)
et_mae_mse = mean_squared_error(y_test, et_mae_predicted)
print("Mean Squared Error of ExtraTreesRegressor "
      "Trained w/ MSE Criterion: {}").format(et_mse_mse)
print("Mean Squared Error of ExtraTreesRegressor "
      "Trained w/ MAE Criterion: {}").format(et_mae_mse)

et_mse_mae = mean_absolute_error(y_test, et_mse_predicted)
et_mae_mae = mean_absolute_error(y_test, et_mae_predicted)
print("Mean Absolute Error of ExtraTreesRegressor "
      "Trained w/ MSE Criterion: {}").format(et_mse_mae)
print("Mean Absolute Error of ExtraTreesRegressor "
      "Trained w/ MAE Criterion: {}").format(et_mae_mae)
print("")
