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

# loading model and evalueating on given set 


list_eval = loadList("data/demo/lists/test.list")

fname_model_weights = "models/demo/%s.h5" % model_ID

model4output.load_weights(fname_model_weights)

errors, example, outputs = eval_dataset(
  list_eval,
  loader,
  model4output,
  model4example,
  funs_eval=len(ys)*[compute_errorMSEDTW]
)

print(errors)
