from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                    boston.target,
                                                    test_size=0.25,
                                                    random_state=0)

params_dict = {"min_impurity_split": [0.2, 0.5, 0.8, 1.1, 1.4,
                                      1.7, 2.0, 2.3, 2.6, 2.9, 3.2]}

x = []

y_num_nodes = []
y_MSE_test = []
y_MSE_train = []

# test default parameters
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X_train, y_train)

# add num nodes to result list
y_num_nodes.append(reg.tree_.node_count)

# get predictions and add to result list
y_predicted_test = reg.predict(X_test)
test_MSE = mean_squared_error(y_test, y_predicted_test)
y_MSE_test.append(test_MSE)
y_predicted_train = reg.predict(X_train)
train_MSE = mean_squared_error(y_train, y_predicted_train)
y_MSE_train.append(train_MSE)

# add a label for default case
x.append(0)

for param_value in params_dict["min_impurity_split"]:
    reg = DecisionTreeRegressor(random_state=0,
                                min_impurity_split=param_value)
    reg.fit(X_train, y_train)

    # add number of nodes to result list
    y_num_nodes.append(reg.tree_.node_count)

    # get predictions and add to result list
    y_predicted_test = reg.predict(X_test)
    test_MSE = mean_squared_error(y_test, y_predicted_test)
    y_MSE_test.append(test_MSE)
    y_predicted_train = reg.predict(X_train)
    train_MSE = mean_squared_error(y_train, y_predicted_train)
    y_MSE_train.append(train_MSE)

    # create label for this bar
    x.append(param_value)

# plot min_impurity_split vs Num Nodes
plt.plot(x, y_num_nodes)
# add some text for label, title and axes ticks
plt.ylabel('Number of Nodes in Tree')
plt.title('min_impurity_split vs Number of Nodes in Trees')
plt.xlabel('min_impurity_split value')
plt.tight_layout()

# plot min_impurity_split vs train_mse
plt.figure()
plt.plot(x, y_MSE_train)
# add some text for labels, title and axes ticks
plt.ylabel('Train Set MSE')
plt.title('min_impurity_split vs Train Set MSE')
plt.xlabel('min_impurity_split value')
plt.tight_layout()

# plot min_impurity_split vs Num Nodes
plt.figure()
plt.plot(x, y_MSE_test)
# add some text for labels, title and axes ticks
plt.ylabel('Test Set MSE')
plt.title('min_impurity_split vs Test Set MSE')
plt.xlabel('min_impurity_split value')
plt.tight_layout()

plt.show()
