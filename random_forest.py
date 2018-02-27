import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

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
print(X)


for col in range(0,5):
  le = preprocessing.LabelEncoder()
  X[:,col] = le.fit_transform(X[:,col])

print(X)

regr = RandomForestRegressor(n_estimators=20, max_depth=4, max_features=5)
regr.fit(X, Y)
pred_Y = regr.predict(X)






