from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import cProfile

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                    boston.target,
                                                    test_size=0.25,
                                                    random_state=0)

# Fit regression model
mse_regressor = DecisionTreeRegressor(random_state=0)
mae_regressor = DecisionTreeRegressor(random_state=0, criterion="mae")

pr = cProfile.Profile()
pr.enable()
mse_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

pr = cProfile.Profile()
pr.enable()
mae_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

mse_predicted = mse_regressor.predict(X_test)
mae_predicted = mae_regressor.predict(X_test)

mse_mse = mean_squared_error(y_test, mse_predicted)
mae_mse = mean_squared_error(y_test, mae_predicted)
print "Mean Squared Error of Tree Trained w/ MSE Criterion: {}".format(mse_mse)
print "Mean Squared Error of Tree Trained w/ MAE Criterion: {}".format(mae_mse)

mse_mae = mean_absolute_error(y_test, mse_predicted)
mae_mae = mean_absolute_error(y_test, mae_predicted)
print "Mean Absolute Error of Tree Trained w/ MSE Criterion: {}".format(mse_mae)
print "Mean Absolute Error of Tree Trained w/ MAE Criterion: {}".format(mae_mae)
