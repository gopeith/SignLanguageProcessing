import h5py
import re
import random
import math

import numpy

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Add, Dropout
from keras import backend as K
from keras.layers import Layer, Input, Bidirectional, LSTM, GRU, Embedding, Multiply, BatchNormalization, Concatenate, Add, Lambda, Subtract
from keras.utils import to_categorical

from modeling import *
from data import *
from training import *
import accumulators
from dtw import dtw, fakedtw
from utils import *

from model_FF_05 import *


# file with the best (on stop set) results
fname_model_weights = "models/%s.h5" % model_ID

# file with an example of the chosen output
fname_example = "temp/%s-example.txt" % model_ID

list_training = loadList("data/training.list")
list_stop = loadList("data/stop.list")


n_files_in_batch = 128
if n_files_in_batch > len(list_training):
  n_files_in_batch = len(list_training) # carefull

# how often is model evaluated using the stop set.
# 1 means after every reestimation 
n_batches = 1

# when stop training
max_steps = 100000

optimizer = accumulators.AdamAccumulate(lr=0.001, decay=0.0, accum_iters=n_files_in_batch)

wMSE = 1.0 # weight for MSE loss
wATT = 1.0 / 98.0 # weight for our attention based loss loss

training(
  model4training=model4training,
  model4output=model4output,
  model4example=model4example,
  datamod4training=synch, 
  loss=len(ys)*[MSE]+len(ys_att)*[lossAttMSE],
  wloss=len(ys)*[wMSE]+len(ys_att)*[wATT],
  loader=loader,
  list_training=list_training,
  list_stop=list_stop,
  mask4print="%.4f",
  optimizer=optimizer,
  testing_interval=n_batches*n_files_in_batch,
  fname_example=fname_example,
  fname_model_weights=fname_model_weights,
  funs_eval=len(ys)*[compute_errorMSEDTW],
  max_steps=max_steps,
)
  
hfTar.close()
