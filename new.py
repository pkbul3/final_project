import numpy as np
from sklearn.linear_model import LogisticRegression

# define training data
X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y_train = np.array([0, 1, 1, 0])

# create logistic regression model and train on data
clf = LogisticRegression(random_state=0, solver='lbfgs')
clf.fit(X_train, y_train)

# define test data
X_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# make predictions on test data
y_pred = clf.predict(X_test)

# print predictions
print(y_pred)
