import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from collections import Counter

def lowercase_key(d:dict):
  return {k.lower():v for k,v in d.items()}

def get_cmap(N):
  color_list = []
  for i in range(N):
    color = np.random.random(3)
    color_list.append(color)
  return color_list
  
def get_name_plot(N):
  name_list = []
  for i in range(N):
    name = 'graph_{}'.format(i+1)
    name_list.append(name)
  return name_list

def plot_loggers(log_list:list, cols_of_log:list):
  assert [type(logger) == Logger and logger.num_cols == len(cols_of_log) for logger in log_list]
  color_map = get_cmap(len(log_list))
  label_name = get_name_plot(len(log_list))
  values = []
  for logger in log_list:
    val = np.array(logger.values).T.tolist()
    values.append(val)
  for col in range(1, len(cols_of_log)):
    plt.figure()
    for log in range(len(log_list)):
      try:
        color = log_list[log].color
      except:
        color = color_map[log]
      try:
        label = log_list[log].log_type
      except:
        label = label_name[log]
      
      plt.plot(values[log][0],values[log][col], label=label, color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(cols_of_log[col])
    plt.legend()
    plt.tight_layout()
    plt.show()
  
class Logger():
  def __init__(self, save_folder:str, file_name:str, cols:list, **kwargs):
    '''
    save_folder: Thư mục lưu file log
    file_name: Tên file log
    cols: List các tên cột (Ví dụ: ['epoch', 'loss', 'score'])
    '''
    self.num_cols = len(cols)
    assert self.num_cols > 0

    if 'log_type' in kwargs:
      self.log_type = kwargs['log_type']
    if 'color' in kwargs:
      self.color = kwargs['color']

    self.cols = [col.lower() for col in cols]
    self.values = []
    
    file_path = os.path.join(save_folder, file_name)
    if os.path.isfile(file_path):
      df = pd.read_csv(file_path, sep=',')
      assert self.cols == list(df.columns), 'Expected cols and cols in file dont match.'
      if df.shape[0] != 0:
        for i in range(df.shape[0]):
          val = []
          for col in self.cols:
            val.append(df[col].iloc[i])
          self.values.append(val)
      self.log_file = open(file_path, "a", buffering=1)
    else:
      self.log_file = open(file_path, "w", buffering=1)
      col_line = ''
      for i in range(self.num_cols):
        col_line += self.cols[i]
        if i != self.num_cols-1:
          col_line += ','
        else:
          col_line += '\n'

      self.log_file.write(col_line)

  def add_val(self, **kwargs):
    '''
    expect kwargs are values refer to cols 
    (eg: epoch=0, loss=9999, score=10e-10)
    '''
    assert len(kwargs) == self.num_cols
    kwargs = lowercase_key(kwargs)
    assert Counter(kwargs.keys()) == Counter(self.cols), 'Cols and args dont fit each other'
    line = ''
    value = []
    for idx, col in enumerate(self.cols):
      value.append(kwargs[col])
      line += str(kwargs[col])
      if idx != self.num_cols-1:
        line += ','
      else:
        line += '\n'
    self.log_file.write(line)
    self.values.append(value)
  
  def add_vals(self, **kwargs):
    '''
    expect kwargs are values refer to cols 
    and each value is a list.
    (eg: epoch=[0,1], loss=[9999,9998], score=[10e-10,10e-9])
    '''
    assert len(kwargs) == self.num_cols
    kwargs = lowercase_key(kwargs)
    assert Counter(kwargs.keys()) == Counter(self.cols), 'Cols and args dont fit each other'

    list_vals = []
    for col in self.cols:
      list_vals.append(list(kwargs[col]))
    list_vals = np.array(list_vals).T.tolist()
    for i in range(len(list_vals)):
      args = {}
      for idx, val in enumerate(list_vals[i]):
        args[self.cols[idx]] = val
      self.add_val(**args)

  def plot(self):
    plot_list = np.array(self.values).T.tolist()
    for i in range(1, self.num_cols):
      plt.plot(plot_list[0], plot_list[i])
    plt.legend(self.cols[1:])
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    try:
      plt.title(self.log_type)
    except:
      plt.title('Learning curves')
    plt.show()

  def __del__(self):
    self.log_file.close()
  