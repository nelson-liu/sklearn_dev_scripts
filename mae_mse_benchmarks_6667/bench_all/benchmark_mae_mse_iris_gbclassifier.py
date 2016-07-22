from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import cProfile

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.25,
                                                    random_state=0)


# Fit GradientBoostingClassifier regression model####################################
gb_mse_regressor = GradientBoostingClassifier(random_state=0, max_leaf_nodes=100, n_estimators=10)
gb_mae_regressor = GradientBoostingClassifier(random_state=0, criterion="mae", max_leaf_nodes=100, n_estimators=10)

print "Profiling Friedman MSE GradientBoostingClassifier"
pr = cProfile.Profile()
pr.enable()
gb_mse_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

print "Profiling MAE GradientBoostingClassifier"
pr = cProfile.Profile()
pr.enable()
gb_mae_regressor.fit(X_train, y_train)
pr.disable()
pr.print_stats(sort='time')

gb_mse_predicted = gb_mse_regressor.predict(X_test)
gb_mae_predicted = gb_mae_regressor.predict(X_test)

gb_mse_mse = mean_squared_error(y_test, gb_mse_predicted)
gb_mae_mse = mean_squared_error(y_test, gb_mae_predicted)
print("Mean Squared Error of GradientBoostingClassifier "
      "Trained w/ Friedman MSE Criterion: {}").format(gb_mse_mse)
print("Mean Squared Error of GradientBoostingClassifier "
      "Trained w/ MAE Criterion: {}").format(gb_mae_mse)

gb_mse_mae = mean_absolute_error(y_test, gb_mse_predicted)
gb_mae_mae = mean_absolute_error(y_test, gb_mae_predicted)
print("Mean Absolute Error of GradientBoostingClassifier "
      "Trained w/ Friedman MSE Criterion: {}").format(gb_mse_mae)
print("Mean Absolute Error of GradientBoostingClassifier "
      "Trained w/ MAE Criterion: {}").format(gb_mae_mae)
print("")
