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

params_dict = {"min_impurity_split": [0.2, 0.5, 3],
               "max_leaf_nodes": [10, 50, 100],
               "max_depth": [3, 5, 7],
               "min_samples_leaf": [3, 5, 10]}

labels = []

total_num_params = sum([len(x) for x in params_dict.values()]) + 1
x = xrange(total_num_params)

y_num_nodes = []
y_MSE = []

# test default parameters
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X_train, y_train)

# add num nodes to result list
y_num_nodes.append(reg.tree_.node_count)

# get predictions and add to result list
y_predicted = reg.predict(X_test)
test_MSE = mean_squared_error(y_test, y_predicted)
y_MSE.append(test_MSE)

# add a label for default case
labels.append("Default Tree")

for param in params_dict.keys():
    for param_value in params_dict[param]:
        reg = DecisionTreeRegressor(random_state=0)
        reg.set_params(**{param: param_value})
        reg.fit(X_train, y_train)

        # add num nodes to result list
        y_num_nodes.append(reg.tree_.node_count)

        # get predictions and add to result list
        y_predicted = reg.predict(X_test)
        test_MSE = mean_squared_error(y_test, y_predicted)
        y_MSE.append(test_MSE)

        # create label for this bar
        labels.append("{}: {}".format(param, param_value))

# plot Parameters vs Num Nodes
plt.bar(x, y_num_nodes, align='center')
# add some text for labels, title and axes ticks
plt.ylabel('Number of Nodes in Tree')
plt.title('Number of Nodes in Trees Constructed With Various Parameters')
plt.xlabel('Tree Parameters')
plt.xticks(x, labels, rotation=45, rotation_mode="anchor", ha="right")
plt.tight_layout()

# plot Parameters vs MSE
plt.figure()
plt.bar(x, y_MSE, align='center')
# add some text for labels, title and axes ticks
plt.ylabel('Mean Squared Error of Predictions')
plt.title('MSE of Trees Constructed With Various Parameters')
plt.xlabel('Tree Parameters')
plt.xticks(x, labels, rotation=45, rotation_mode="anchor", ha="right")
plt.tight_layout()

plt.show()
