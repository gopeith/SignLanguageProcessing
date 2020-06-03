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
from dtw import dtw, fakedtw
from utils import *
from training import *


# filenames for weights, etc. are derived from this ID  
model_ID = "FF_20200603"

dtype = "float32"

n_steps = 7 # number of frames for each word
dim_y = 98 # ouput size
dim_emb = 256 # embeddings size
dim_rep = 1024 # "repository" size
dim_h = 1024 # hidden layers size

if False:
  # avoiding using multiple CPUs (not always succesfully) and memory leaks in tensorflow
  conf = K.tf.ConfigProto(
    device_count={'CPU': 1},
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
  )
  K.set_session(K.tf.Session(config=conf))

# make experiment deterministic
random.seed(123)
numpy.random.seed(123)  

# This function returns a dictionary with inputs and targets.
# Keys "inputs" and "targets" are mandatory. 
# Values must be lists not tuples. The lists will be appended and modified when synchronization is performed.
def loader(key):
  video = numpy.array(hfTar.get(key))
  text_int = text[key]
  text_float = to_categorical(text_int, num_classes=dim_text)
  
  return {
    "inputs": [
      text_float,
      info_spk[key],
      #info_today[key],
    ],
    "targets": [
      video for y in ys + ys_att
    ],
    "key": key,
  }


# A function for tied layers.
def get_dense(Lname, dim, stop, activation=lambda x:x, use_bias=True, init_weights=None, trainable=True):
  # create or use already created layer
  if not Lname in Ls:
    if use_bias:
      weights=[ws.pop(0), ws.pop(0)] if len(ws) > 0 else init_weights
    else:
      weights=[ws.pop(0)] if len(ws) > 0 else init_weights    
    Ls[Lname] = Dense2(dim, activation=activation, use_bias=use_bias, weights=weights, trainable=trainable)
  L = Ls[Lname]
  L.stop = stop
  return L


# "embedding" layer that code input language A
def embA(x, stop=False, mod=""):
  x = ConcatenateMatrixAndVectors()([x] + addinf)
  L = get_dense("Aemb" + mod, dim_emb, stop, use_bias=False)
  x = L(x)
  x = LayerNormalization()(x)
  return x


def translation_from_A_to_B(x, stop=False, mod="", Nh=0):

  for i in range(Nh):
    x_old = x
    x = Splicing([-1, 0, 1])(x)
    x = ConcatenateMatrixAndVectors()([x] + addinf)
    L = get_dense(("A_a[%d]" % i) + mod, dim_h, stop, activation=keras.activations.relu, use_bias=True)
    x = L(x)
    x = Dropout(rate=pDropout)(x)
    L = get_dense(("A_b[%d]" % i) + mod, dim_emb, stop, use_bias=True)
    x = L(x)
    x = Add()([x, x_old])
    x = LayerNormalization()(x)

  return x


# "deembedding" layer that decode output, ie. language B
def invEmbB(x, stop=False, mod=""):
  x = ConcatenateMatrixAndVectors()([x] + addinf)
  L = get_dense("An_a" + mod, n_steps * dim_emb, stop, use_bias=True)
  x = L(x)
  x = MakeSteps2(nSteps=n_steps, dimy=dim_emb)(x)
  x = Splicing([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])(x)
  x = ConcatenateMatrixAndVectors()([x] + addinf)
  L = get_dense("An_b" + mod, dim_rep, stop, activation=keras.activations.softmax, use_bias=True)
  x = L(x)
  L = get_dense("An_c" + mod, dim_y, stop, use_bias=False, init_weights=[rep_values], trainable=False)
  x = L(x)
  #x = ConstantBias(mu_values)(x)
  return x


# dictionary for computed DTW synchronization
saved_paths = {}

# this function synchronizes outputs and targets
# The synchronization is not provided every gradient computation but only 30% of time.
# 70% of time the function reuses synchronization save in the dictionary saved_paths  
def synch(data):
  if data["key"] in saved_paths:
    paths = saved_paths[data["key"]]
    used_saved_paths = (random.randint(0, 100) <= 30)
  else:
    paths = [None for y_synch in ys_synch]
    used_saved_paths = True

  used_saved_paths = True

  if not used_saved_paths:
    inputs_values = data["inputs"]
    inputs_values = [addDim(x) for x in inputs_values] # 2D -> 3D
    ys_values = model4synch.predict_on_batch(inputs_values)
    if not type(ys_values) is list:
      ys_values = [ys_values]
  else:
    T_y = n_steps * data["inputs"][0].shape[0]
    ys_values = [numpy.zeros((T_y, 1), dtype=dtype) for y_synch in ys_synch]

  for idx in range(len(ys_values)):
    i_out = idx
    i_tar = idx
    y_values = ys_values[i_out]
    tar_values = data["targets"][i_tar]
    
    if used_saved_paths:
      if paths[idx] is None:
        d, path_y, path_tar = fakedtw(
          y_values,
          tar_values,
          0.3,
        )
      else:
        d, path_y, path_tar = paths[idx]
    else:
      d, path_y, path_tar = dtw(
        y_values,
        tar_values,
        0.3,
      )

    paths[idx] = (d, path_y, path_tar)

    data["inputs"].append(numpy.asarray(path_y, dtype="int64"))
    data["targets"][i_tar] = data["targets"][i_tar][path_tar]
  
  saved_paths[data["key"]] = paths 

  return data


text, dim_text = loadText("data/text")

print("dim_text = %d" % dim_text)

hfTar = h5py.File("data/keypoints-filt-deltas-norm.h5", "r")

info_spk, dim_spk = loadInfo("data/info_spk", dtype=dtype)
#info_today, dim_today = loadInfo("data/info_today", dtype=dtype)

reg = keras.regularizers.l2(0.001)

input_text = Input(batch_shape=(None, None, dim_text), dtype=dtype, name="textf")
input_spk = Input(batch_shape=(None, None, dim_spk), dtype=dtype, name="spk")
#input_today = Input(batch_shape=(None, None, dim_today), dtype=dtype, name="today")

inputs = [input_text, input_spk]#, input_today]

# reshaping 3D inputs into 2D arrays
x_text = Lambda(lambda x: reshape_2d_fixed_dim(x, dim_text))(input_text)
x_spk = Lambda(lambda x: reshape_2d_fixed_dim(x, dim_spk))(input_spk)
#x_today = Lambda(lambda x: reshape_2d_fixed_dim(x, dim_today))(input_today)
addinf = [x_spk]#, x_today]


# ws is a list of weights that is important when a new layer is added etc.
#fname_weights = "models/FF_20200603-clear.h5"
fname_weights = None
if fname_weights is None:
  # empty list means random initialization
  ws = []
else:
  # load weight from file
  ws = loadAllDenses2(fname_weights)

# dropout rate
pDropout = 0.1

# mean of outputs 
mu_values = loadMatrixFloatText("temp/mu", dtype=dtype)
rep_values = loadMatrixFloatText("temp/repository-%d" % dim_rep, dtype=dtype)

# dictionary of layers. Important for tied layers.
Ls = {}

# following model translates from language A into language B

a = x_text
ae = embA(a)
ybe_0 = ae
ybe_1 = translation_from_A_to_B(ybe_0, Nh=1, mod="1")
#ybe_2 = translation_from_A_to_B(ybe_1, Nh=1, mod="2")
#ybe_3 = translation_from_A_to_B(ybe_2, Nh=1, mod="3")

#yb_0 = invEmbB(ybe_0, False, mod="0")
yb_1 = invEmbB(ybe_1, False, mod="1")
#yb_2 = invEmbB(ybe_2, False, mod="2")
#yb_3 = invEmbB(ybe_3, False, mod="3")

# outputs
ys = [
  #yb_3,
  #yb_2,
  yb_1,
  #yb_0,
]

# outputs trained by means of an attention mechanism
ys_att = [
  yb_1,
]

# outputs which will be synchronized
ys_synch = ys

inputs_dtw = [Input(batch_shape=(None, None), dtype="int64", name="dtw%d" % i) for i in range(len(ys))]

# special layer synchronizing output (targers are syncronized outside models)
ys_dtw = [UseDTW()([ys[i], inputs_dtw[i]]) for i in range(len(ys_synch))]

# back into the 3D shapes
make_it_3d = Lambda(lambda x: reshape_3d(x))
ys_3d = [make_it_3d(y) for y in ys]
ys_dtw_3d = [make_it_3d(y) for y in ys_dtw]
ys_att_3d = [make_it_3d(y) for y in ys_att]

# these outpus will be trained
model4training = Model(inputs=inputs+inputs_dtw, outputs=ys_dtw_3d+ys_att_3d)

# these outputs will be evalueated
model4output = Model(inputs=inputs, outputs=ys)

# these outputs serves for DTW synchronization
model4synch = Model(inputs=inputs, outputs=ys_synch)

# this output will be saved for monitoring
model4example = Model(inputs=inputs, outputs=[ys[0]])
