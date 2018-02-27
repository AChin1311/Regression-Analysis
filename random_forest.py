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

def load_dataset():
  data_x = []
  data_y = []
  days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  with open("network_backup_dataset.csv", "r") as csvfile:
    content = csv.reader(csvfile)
    next(content, None) # skip the headers

    for row in content:
      week, day, start_time, workflow_ID, filename, backup_size, backup_time = row
      # tmp = []
      # tmp.append(int(week))
      # tmp.append(days.index(day)+1)
      # tmp.append(int(start_time))
      # tmp.append(int(workflow_ID[-1]))
      # tmp.append(int(filename[5:]))
      data_x.append(row[:5])
      # data_x.append(tmp)
      data_y.append(float(backup_size))
    print(len(data_y))
    print(len(data_x), len(data_x[0]))
    
  csvfile.close()
  return np.array(data_x), np.array(data_y)

X, Y = load_dataset()

# encode catigorical labels to numerical labels
for col in range(0,5):
  le = preprocessing.LabelEncoder()
  X[:,col] = le.fit_transform(X[:,col])


num_trees = list(range(1,201))
num_features = [5]
rmse = []
oob_errors = []


for ntrees in num_trees:
  for nfeatures in num_features:
    print("tree: ", ntrees, ", features: ", nfeatures)
    
    # OOB (out of bag) error validation
    regr = RandomForestRegressor(n_estimators=ntrees, max_depth=4, max_features= nfeatures, oob_score=True, n_jobs=-1)
    regr.fit(X, Y)
    oob_errors.append(1- regr.oob_score_)
    
    # RMSE (cross validation)
    
    # mse_scorer = make_scorer(mean_squared_error, greater_is_better=True)
    # rmse.append(np.sqrt(np.mean(cross_validation.cross_val_score(regr, X, Y, cv = 10, scoring=mse_scorer, n_jobs=-1))))
    
    mse =[]
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
      X_train, X_test = X[train_index], X[test_index]
      Y_train, Y_test = Y[train_index], Y[test_index]
      regr = RandomForestRegressor(n_estimators=ntrees, max_depth=4, max_features= nfeatures, oob_score=True, n_jobs=-1)
      regr.fit(X_train, Y_train)
      Y_predict = regr.predict(X_test)
      mse.append(mean_squared_error(Y_test, Y_predict))
    rmse.append(np.sqrt(np.mean(mse)))

# TODO: modify the plotting code below into loop above

plt.figure()
y = oob_errors
x = num_trees
plt.plot(x, y)
plt.savefig('plot/figure1.png')
plt.clf()


plt.figure()
y = rmse
x = num_trees
plt.plot(x, y)
plt.savefig('plot/figure2.png')
plt.clf()