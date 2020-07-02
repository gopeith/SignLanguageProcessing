import h5py
import re
import random
import os
import sys
from datetime import datetime 

import numpy
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import backend as K
from keras.layers import Layer, Input
from keras.models import model_from_json

from utils import *


def make_dir_for_file(fname):
  path = re.sub(r"[^\\/]+$", "", fname)
  try:  
    os.mkdir(path)
  except OSError:  
    #print("Cannot create the directory %s" % path) 
    pass; # Perhaps, the path exists.


def save(fname_weights, model):
  #make_dir_for_file(fname_json)
  make_dir_for_file(fname_weights)
  # serialize model to JSON
  #model_json = model.to_json()
  #with open(fname_json, "w") as json_file:
  #  json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(fname_weights)


def addDim(M):
  if len(M.shape) == 1:
    return M.reshape((1, M.shape[0]))
  return M.reshape((1, M.shape[0], M.shape[1]))
  

def eval_dataset(list_eval, loader, model4output, model4example, funs_eval):
  sum0 = None
  sum1 = None
  example = None
  outputs = {}
  
  for key in list_eval:
    data = loader(key)
    inputs_values = data["inputs"]
    inputs_values = [addDim(x) for x in inputs_values] # 2D -> 3D
    ys_values = model4output.predict_on_batch(inputs_values)
    if not ((type(ys_values) is list) or (type(ys_values) is tuple)):
      ys_values = [ys_values]
   
    if sum0 is None:
      sum0 = [0 for foo in range(len(ys_values))]
      sum1 = [0.0 for foo in range(len(ys_values))]
    
    if example is None:
      if not model4example is None:
        example = model4example.predict_on_batch(inputs_values)
    
    for i in range(len(ys_values)):
      y_values = ys_values[i]
      tar_values = data["targets"][i]
      fun_eval = funs_eval[i]
      e, m, n = fun_eval(y_values, tar_values)
      sum0[i] = sum0[i] + m * n
      sum1[i] = sum1[i] + e
      if i == 0:
        outputs[key] = y_values
  errors = []
  for i in range(len(sum0)):
    errors.append(sum1[i] / sum0[i])
  return errors, example, outputs


def prepare_data(key, loader, model4output, datamod): 
  data = loader(key)
  data = datamod(data)
  inputs_values = data["inputs"]
  inputs_values = [addDim(x) for x in inputs_values] # 2D -> 3D 
  tar_values = data["targets"]
  tar_values = [addDim(tar) for tar in tar_values] # 2D -> 3D
  return inputs_values, tar_values


def training(
  model4training,
  model4output,
  model4example,
  loss,
  loader,
  list_training,
  list_stop,
  optimizer,
  funs_eval,
  testing_interval,
  fname_model_weights=None,
  fname_example=None,
  list_test=None,
  mask4print="%e",
  wloss=None,
  max_steps=None,
  datamod4training=lambda data: data,
  loader_test=None
):

  if loader_test is None:
    loader_test = loader

  if not fname_example is None:
    make_dir_for_file(fname_example)

  model4training.compile(loss=loss, loss_weights=wloss, optimizer=optimizer)
  
  random.shuffle(list_training)
  
  e_stop_min = None
  e_test_min = None  
  i_step = 0
  outputs_test = None
  while True:

    # testing
    if i_step % testing_interval == 0:
      print("")  
      print("testing (%d steps):" % i_step)  
      print("")  
      es_stop, example_stop, outputs_stop = eval_dataset(list_stop, loader_test, model4output, model4example, funs_eval)
      e_stop = es_stop[0]
      if not fname_example is None:
        saveMatrixText(fname_example, example_stop) 
      if e_stop_min is None or e_stop_min >= e_stop:
        e_stop_min = e_stop
        if not list_test is None:
          es_test, example_test, outputs_test = eval_dataset(list_test, loader_test, model4output, model4example, funs_eval)
          e_test = es_test[0] 
          e_test_min = e_test
        if not fname_model_weights is None:
          try:
            save(fname_model_weights, model4output)
          except OSError:
            print("OSError")
            pass
      print("  actual results:")  
      print("")  
      print("    e(stop) = %s" % "\t".join([mask4print % foo for foo in es_stop]))
      if not list_test is None:
        print("    e(test) = %s" % "\t".join([mask4print % foo for foo in es_test]))
      print("")  
      print("  the best results:")  
      print("")  
      print(("    e(stop) = " + mask4print) % e_stop_min)
      if not list_test is None:
        print(("    e(test) = " + mask4print) % e_test_min)
      print("")
      sys.stdout.flush()

    i_key = i_step % len(list_training)  
    key = list_training[i_key]
    inputs_values, tar_values = prepare_data(key, loader, model4output, datamod4training)

    model4training.fit(inputs_values, tar_values, epochs=1, batch_size=1, verbose=0)
    
    i_step = i_step + 1
    
    if (not max_steps is None) and (i_step >= max_steps):
      break
  
  print("OK")  
  
