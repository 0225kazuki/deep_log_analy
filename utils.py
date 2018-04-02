#!/usr/bin/python
# coding: UTF-8

import pandas as pd
import numpy as np
import pickle
import datetime
import collections
import matplotlib.pyplot as plt
import json


def dump2path(dump_name, rp=False):
#   prefix="./nofilter/"
  if rp:
    prefix="./datasets/rplinear_oneday/"
  else:
    prefix="./datasets/nofilter_oneday/"
  ext=".dump"
  lt_id = int(dump_name.split("_")[0])
  if lt_id < 500:
    prefix += "0000-0499/"
  elif lt_id < 1000:
    prefix += "0500-0999/"
  elif lt_id < 1500:
    prefix += "1000-1499/"
  else:
    prefix += "1500-1999/"
  return prefix + dump_name + ext


def path2dump(path):
  return path.split("/")[-1].split(".dump")[0]


def load_dump(dump_name, rp=False):
  if ".dump" not in dump_name:
    path = dump2path(dump_name, rp=rp); print(path)
    with open(path, "rb") as f:
      obj = pickle.load(f, encoding="bytes")
    return obj
  else:
    with open(dump_name, "rb") as f:
      obj = pickle.load(f, encoding="bytes")
    return obj


def load_dump2df(dump_name, bin_size="sec", rp=False):
  obj = load_dump(dump_name, rp=rp)

  if bin_size=="sec":
    bin_length = 86400
    metrics = lambda x: x.hour*3600 + x.minute*60 + x.second
  elif bin_size=="min":
    bin_length = 1440
    metrics = lambda x: x.hour*60 + x.minute
  else:
    raise("bin_size error")

  all_data = pd.DataFrame(index=np.arange(0, bin_length))
  for d in pd.date_range('20120101', '20130331'):
    tmp_ts = [metrics(i) for i in obj if i.date() == d.date()]
    sec_ts = pd.Series([0]*bin_length, index=np.arange(0, bin_length))
    for k,v in collections.Counter(tmp_ts).items():
        sec_ts[k] = v
    all_data[d] = sec_ts
  return all_data


def make_dataset(dump_names, file_name, bin_size="sec", rp=False):
  x_train = []
  x_labels = []
  x_date_labels = []

  for ev in dump_names:
    print("processing:", ev)
  
    data = load_dump2df(ev, bin_size="min", rp=rp)
      
    for i in data.iteritems():
      x_labels.append(ev)
      x_train.append(np.asarray(i[1], dtype=np.float32))
    x_date_labels.extend([d for d in pd.date_range('20120101', '20130331')])
      
  x_train = np.array(x_train)
  
  print("x_train:", x_train.shape)
  print("x_labels:", len(x_labels))
  print("dump as ",file_name)
  with open(file_name, "wb") as f:
    pickle.dump([x_train, x_labels, x_date_labels], f)


def plot_day(dump_name, date="all", bin_size="sec", rp=False,  description=False):
  if "/" in dump_name:
    obj = load_dump(dump_name)
  else:
    obj = load_dump(dump2path(dump_name, rp=rp))

  if date == "all":
    plot_data = [row.time() for row in obj]
  else:
    plot_year = int(date[:4])
    plot_month = int(date[4:6])
    plot_day = int(date[6:8])

    plot_date = datetime.date(plot_year, plot_month, plot_day)
    plot_data = [row for row in obj if row.date()==plot_date]
    plot_data = [row.time() for row in plot_data]
  plot_data_coll = collections.Counter(plot_data)
    
  if len(plot_data) == 0:
    print("No Data")
    return


  if bin_size == "sec": 
    x = [row.hour*3600 + row.minute*60 + row.second for row in sorted(set(plot_data))]
  elif bin_size == "min":
    x = [row.hour*60 + row.minute for row in sorted(set(plot_data))]
  else:
    raise("bin_size error")

  if description:
    mean = np.mean(np.array(x))
    std = np.std(np.array(x))
    coef_var = std/mean
    kurt = np.sum((np.array(x) - mean)**4)/len(x) / (std**2) - 3

  y = [0]
  for row in sorted(plot_data_coll.items(),key=lambda z:z[0]):
      y.append(row[1]+y[-1])
  y = y[1:]

  # 階段状にする処理
  x = np.sort(np.append(x,x))[1:]
  x = np.insert(x,0,x[0])

  tmp = []
  for row in y:
      tmp.append(row)
      tmp.append(row)

  y = tmp[:-1]
  y = [0] + y

  # plot
  fig = plt.figure(figsize=(10, 5))
  # default left : 0.125　right : 0.9　bottom : 0.1　top : 0.9　wspace : 0.2　hspace : 0.2
  fig.subplots_adjust(left=0.03, right=0.999)
  plt.title(dump_name+"\t"+date)
  plt.plot(x, y)
  if bin_size=="sec":
    plt.xticks([i*3600 for i in range(25)],
               [str(i).zfill(2)+':00\n{0}'.format(i*3600) for i in range(25)],
               fontsize=10,
               rotation=90)
    plt.xlim(0, 86400)
  elif bin_size=="min":
    plt.xticks([i*60 for i in range(25)],
               [str(i).zfill(2)+':00\n{0}'.format(i*60) for i in range(25)],
               fontsize=10,
               rotation=90)
    plt.xlim(0, 1440)
  plt.yticks(fontsize=10)

  plt.grid()

  # plt.savefig(dump_name.split('/')[-1].split('.')[0]+'_'+date+'.png')
  plt.show()

  if description:
    # show statistical data
    print("mean:\t",mean)
    print("std :\t",std)
    print("coef:\t",coef_var)
    print("kurt:\t",kurt)


def get_burst_date(ltid):
    '''
    return datetime.date list
    '''
    
    try:
        with open("burst_df.json", "r") as f:
            burst = json.load(f)[ltid]
    except:
        burst = dict()

    burst_date = []
    for ts, data in burst.items():
        if data != None:
            d = datetime.datetime.fromtimestamp(int(int(ts)/1000))
            burst_date.append(d.date())

    return burst_date
