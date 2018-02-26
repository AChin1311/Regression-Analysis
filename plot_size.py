import csv
import matplotlib.pyplot as plt

def load_dataset(untilweek, untilday):
  data = []
  today = 'Monday'
  index = 0
  data.append([0]*5)
  with open("network_backup_dataset.csv", "r") as csvfile:
    content = csv.reader(csvfile)
    next(content, None) # skip the headers
    for row in content:
      week, day, start_time, workflow_ID, filename, backup_size, backup_time = row
      
      if week == untilweek and day == untilday:
        break
      if day != today:
        today = day
        index += 1
        data.append([0]*5)
      
      workflow_ID = int(workflow_ID[-1])
      data[index][workflow_ID] += float(backup_size)
    print(len(data)) 
  csvfile.close()
  return data

day20 = load_dataset("3", "Sunday")
for i in range(5):
  y = [day20[j][i] for j in range(20)]
  x = range(20)
  plt.plot(x, y)
plt.savefig('plot/day20.png')
plt.clf()

day105 = load_dataset(8, "Monday")
for i in range(5):
  y = [day105[j][i] for j in range(105)]
  x = range(105)
  plt.plot(x, y)
plt.savefig('plot/day105.png')
plt.clf()
