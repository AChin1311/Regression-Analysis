import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import make_scorer, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.tree import export_graphviz
import random

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
    print(len(data_x))
    print(len(data_y))
  csvfile.close()
  return np.array(data_x), np.array(data_y)

X, Y = load_dataset()

# encode catigorical labels to numerical labels
for col in range(0,5):
  le = preprocessing.LabelEncoder()
  X[:,col] = le.fit_transform(X[:,col])

num_trees = list(range(1,201))
num_features = list(range(1,6))

### for 2b-(ii) ###
c = 0.1
min_oob_index = []
plt.figure()
for nfeatures in num_features:
  oob_errors = []
  for ntrees in num_trees:
    print("Doing OOB - features: ", nfeatures, ", trees: ", ntrees)
    # OOB (out of bag) error validation
    regr = RandomForestRegressor(n_estimators=ntrees, max_depth=4, max_features= nfeatures, oob_score=True, n_jobs=-1)
    regr.fit(X, Y)
    oob_errors.append(1- regr.oob_score_)
  
  min_oob_index.append(oob_errors.index(min(oob_errors)))

  y = oob_errors
  x = num_trees
  plt.plot(x, y, lw=2, label= "# of feature = "+str(nfeatures))
  plt.grid(color=str(c), linestyle='--', linewidth=1)
  c = c + 0.1
plt.xlabel('# of trees')
plt.ylabel('OOB')
plt.legend()
plt.savefig('plot/2b(ii)-OOB.png')
plt.clf()

### for 2b-(ii) ###
c = 0.1
min_rmse_index = []
plt.figure()
for nfeatures in num_features:
  rmse = []
  for ntrees in num_trees:  
    print("Doing rmse - features: ", nfeatures, ", trees: ", ntrees)  
    # RMSE (cross validation)
    # mse_scorer = make_scorer(mean_squared_error, greater_is_better=True)
    # rmse.append(np.sqrt(np.mean(cross_validation.cross_val_score(regr, X, Y, cv = 10, scoring=mse_scorer, n_jobs=-1))))
    test_mse =[]
    train_mse =[]
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
      X_train, X_test = X[train_index], X[test_index]
      Y_train, Y_test = Y[train_index], Y[test_index]
      train_size = X_train.shape[0]
      test_size = X_test.shape[0]
      regr = RandomForestRegressor(n_estimators=ntrees, max_depth=4, max_features= nfeatures, n_jobs=-1)
      regr.fit(X_train, Y_train)
      Y_test_predict = regr.predict(X_test)
      Y_train_predict = regr.predict(X_train)
      test_mse.append(mean_squared_error(Y_test, Y_test_predict))
      train_mse.append(mean_squared_error(Y_train, Y_train_predict)) 
    rmse.append(np.sqrt(np.mean(test_mse)))

    if nfeatures == 5 and ntrees == 200:
      print('2b(i): training rmse is ', np.sqrt(np.mean(train_mse)))
      print('2b(i): testing rmse is ', np.sqrt(np.mean(test_mse)))

  min_rmse_index.append(rmse.index(min(rmse)))

  y = rmse
  x = num_trees
  plt.plot(x, y, lw=2, label="# of feature = "+str(nfeatures))
  plt.grid(color=str(c), linestyle='--', linewidth=1)
  c = c + 0.1
plt.xlabel('# of trees')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('plot/2b(ii)-RMSE.png')
plt.clf()

print("min OOB error index:")
print(min_oob_index)
print("min RMSE index:")
print(min_rmse_index)


###  for 2b-(iv)&(v)  ###
# ****** TODO: Choose the best paramenters ****** #
# Web visualization - http://webgraphviz.com/
n = 20
f = 5
regr = RandomForestRegressor(n_estimators=n, max_depth=4, max_features= f, n_jobs=-1)
regr.fit(X, Y)
Y_predict = regr.predict(X)
print("feature_importance: ", regr.feature_importances_)
export_graphviz(regr.estimators_[random.randint(0,n-1)], out_file='tree.dot')

###  for the stupid spec ###

plt.figure()


xi = range(0,2)

yi = [i for i in xi]

plt.figure()
y = Y_predict
x = Y




plt.scatter(x, y, s = 1,  alpha=0.01)

plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('true value')
plt.ylabel('fitted value')
plt.plot(xi,yi)
plt.savefig('plot/2b-Fitted-True.png')
plt.clf()

plt.figure()
y = np.abs(Y_predict-Y)
x = Y_predict
plt.scatter(x, y, s=1, alpha=0.01)
plt.axis([-0.03,1.03,-0.03,1.03])
plt.xlabel('fitted value')
plt.ylabel('residuals')
plt.savefig('plot/2b-Residual-Fitted.png')
plt.clf()




