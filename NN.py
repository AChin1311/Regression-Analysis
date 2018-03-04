import csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import make_scorer, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn import datasets

def load_dataset():
  data_x = []
  data_y = []
  days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  with open("network_backup_dataset.csv", "r") as csvfile:
    content = csv.reader(csvfile)
    next(content, None) # skip the headers

    for row in content:
      week, day, start_time, workflow_ID, filename, backup_size, backup_time = row
      data_x.append(row[:5])
      data_y.append(float(backup_size))
    print(len(data_y))
    print(len(data_x), len(data_x[0]))
    
  csvfile.close()
  return np.array(data_x), np.array(data_y)

X_, Y = load_dataset()
for col in range(0,5):
  le = preprocessing.LabelEncoder()
  X_[:,col] = le.fit_transform(X_[:,col])

enc = OneHotEncoder()
X = enc.fit_transform(X_).toarray()

print(X[1])


# N = list(range(1,101))
# plt.figure()
# activation = ['logistic', 'tanh', 'relu']
# for act in activation:
#   rmse = []
#   for n in N:
#     print("N: ", n)
#     # RMSE (cross validation)
    
#     # mse_scorer = make_scorer(mean_squared_error, greater_is_better=True)
#     # rmse.append(np.sqrt(np.mean(cross_validation.cross_val_score(regr, X, Y, cv = 10, scoring=mse_scorer, n_jobs=-1))))
    
#     mse =[]
#     kf = KFold(n_splits=10, random_state=None, shuffle=False)
#     for train_index, test_index in kf.split(X):
#       X_train, X_test = X[train_index], X[test_index]
#       Y_train, Y_test = Y[train_index], Y[test_index]

#       clf = MLPRegressor(hidden_layer_sizes=(n), activation=act)

#       clf.fit(X_train, Y_train)
#       Y_predict = clf.predict(X_test)
#       mse.append(mean_squared_error(Y_test, Y_predict))
#     rmse.append(np.sqrt(np.mean(mse)))

#   print("min_index:", act, rmse.index(min(rmse)))
#   # TODO: modify the plotting code below into loop above

#   y = rmse
#   x = N
#   plt.plot(x, y, lw=2, label=act)
# plt.xlabel('# of hidden units')
# plt.ylabel('RMSE')
# plt.legend()
# plt.savefig('plot/NN.png')
# plt.clf()

# train rmse and test rmse
train_mse =[]
test_mse =[]
train_rmse =[]
test_rmse =[]
n = 85
act = 'relu'
kf = KFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  Y_train, Y_test = Y[train_index], Y[test_index]
  clf = MLPRegressor(hidden_layer_sizes=n, activation=act)
  clf.fit(X_train, Y_train)
  Y_predict = clf.predict(X)
  Y_test_predict = clf.predict(X_test)
  Y_train_predict = clf.predict(X_train)
  train_mse.append(mean_squared_error(Y_train, Y_train_predict))
  test_mse.append(mean_squared_error(Y_test, Y_test_predict))
train_rmse.append(np.sqrt(np.mean(train_mse)))
test_rmse.append(np.sqrt(np.mean(test_mse)))
print('train_rmse = ', train_rmse[0])
print('test_rmse = ', test_rmse[0])


# plotting fitted vs true value
plt.figure()
xi = range(0,2)
yi = [i for i in xi]
y = Y_predict
x = Y
plt.scatter(x, y, s = 1,  alpha=0.01)
plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('true value')
plt.ylabel('fitted value')
plt.plot(xi,yi)
plt.savefig('plot/2c-Fitted-True.png')
plt.clf()

plt.figure()
y = np.abs(Y_predict-Y)
x = Y_predict
plt.scatter(x, y, s=1, alpha=0.01)
plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('fitted value')
plt.ylabel('residuals')
plt.savefig('plot/2c-Residual-Fitted.png')
plt.clf()


