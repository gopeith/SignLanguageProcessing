import math

import numpy

import tensorflow as tf

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Layer, Input, concatenate
from keras import backend as K

from dtw import dtw


def applyTimeShift(x, shift):
  xShape = K.shape(x)
  nMissing = K.minimum(abs(shift), xShape[0])
  if shift == 0:
    y = x
  if shift > 0:
    yMain = x[shift:xShape[0], 0:xShape[1]]
    xT = x[(xShape[0] - 1):xShape[0], 0:xShape[1]]
    yAppendix = K.tile(xT, (nMissing, 1))
    y = K.concatenate([yMain, yAppendix], axis=0)
  if shift < 0:
    yMain = x[0:K.maximum((xShape[0] + shift), 0), 0:xShape[1]]
    x0 = x[0:1, 0:xShape[1]]
    yAppendix = K.tile(x0, (nMissing, 1))
    y = K.concatenate([yAppendix, yMain], axis=0)
  return y


def applyTimeShifts(x, shifts, weights=None):
  y = []
  i_weight = 0
  for shift in shifts:
    if weights is None:
      y.append(applyTimeShift(x, shift))    
    else:
      y.append(weights[i_weight] * applyTimeShift(x, shift))
    i_weight = i_weight + 1
  y = K.concatenate(y, axis=1)
  return y


class Splicing(Layer):

  def __init__(self, shifts, shiftWeights=None, **kwargs):
    self.shifts = shifts
    self.shiftWeights = shiftWeights
    super(Splicing, self).__init__(trainable=False, **kwargs)

  def build(self, input_shape):
    super(Splicing, self).build(input_shape)

  def call(self, x):
    return applyTimeShifts(x, self.shifts, self.shiftWeights)

  def compute_output_shape(self, inputs_shape):
    return (None, len(self.shifts) * inputs_shape[1])


class ConcatenateMatrixAndVectors(Layer):

  def __init__(self, **kwargs):
    super(ConcatenateMatrixAndVectors, self).__init__(trainable=False, **kwargs)

  def build(self, input_shape):
    super(ConcatenateMatrixAndVectors, self).build(input_shape)

  def call(self, x):
    m = x[0]
    mShape = K.shape(m)
    vs = x[1:len(x)]
    y = [m]
    for v in vs:
      vRep = K.tile(v, (mShape[0], 1))
      y.append(vRep)    
    y = K.concatenate(y, axis=1)
    return y

  def compute_output_shape(self, inputs_shape):
    T = inputs_shape[0][0]
    dim = inputs_shape[0][1]
    for i in range(1, len(inputs_shape)):
       dim = dim + inputs_shape[i][1]
    return (T, dim)


class Dense2(Layer):

  def __init__(self, output_dim, use_bias=False, activation=lambda x:x, trainable=True, **kwargs):
    self.output_dim = output_dim
    self.activation = activation
    self.stop = False
    self.use_bias = use_bias
    self.trainable = trainable
    super(Dense2, self).__init__(**kwargs)

  def build(self, input_shape):
    self.kernel = self.add_weight(name='kernel', 
      shape=(input_shape[1], self.output_dim),
      initializer='uniform',
      trainable=self.trainable)
    if self.use_bias:
      self.bias = self.add_weight(name='bias', 
        shape=(self.output_dim, ),
        initializer='zeros',
        trainable=self.trainable)
    super(Dense2, self).build(input_shape)

  def call(self, x):
    if self.stop:
      kernel = K.stop_gradient(self.kernel)
    else:
      kernel = self.kernel
    if self.use_bias:
      if self.stop:
        b = K.stop_gradient(self.bias)
      else:
        b = self.bias
    else:
      b = 0
    return self.activation(K.dot(x, kernel) + b)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)


class LayerNormalization(Layer):

  def __init__(self, eps=1e-6, **kwargs):
    self.eps = eps
    super(LayerNormalization, self).__init__(**kwargs)

  def build(self, input_shape):
    #self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
    #                             initializer=Ones(), trainable=True)
    #self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
    #                            initializer=Zeros(), trainable=True)
    super(LayerNormalization, self).build(input_shape)

  def call(self, x):
    mean = K.mean(x, axis=-1, keepdims=True)
    std = K.std(x, axis=-1, keepdims=True)
    #return self.gamma * (x - mean) / (std + self.eps) + self.beta
    return (x - mean) / (std + self.eps)

  def compute_output_shape(self, input_shape):
    return input_shape


class MakeSteps2(Layer):

  def __init__(self, nSteps, dimy, **kwargs):
    self.nSteps = nSteps
    self.dimy = dimy
    super(MakeSteps2, self).__init__(trainable=False, **kwargs)

  def build(self, input_shape):
    super(MakeSteps2, self).build(input_shape)

  def call(self, x):
    dtype = x.dtype
    nSteps = self.nSteps
    xShape = K.shape(x)
    #xShape = K.print_tensor(xShape, "xShape")
    dim_y = self.dimy #K.cast(xShape[1] / nSteps, dtype="int32")
    O = K.zeros((1, dim_y), dtype=dtype)
    I = K.ones((1, dim_y), dtype=dtype)
    y = 0
    for i in range(nSteps):
      x_i = x[0:xShape[0], (i * dim_y):((i + 1) * dim_y)]
      x_i = K.repeat_elements(x_i, nSteps, axis=0)
      m = []
      for j in range(nSteps):
        if j == i:
          m.append(I)
        else:
          m.append(O)
      m = K.concatenate(m, axis=0)
      m = K.tile(m, (xShape[0], 1))
      y = y + m * x_i
    return y

  def compute_output_shape(self, inputs_shape):
    nSteps = self.nSteps
    dim_y = self.dimy
    return (None, dim_y)


class ConstantBias(Layer):

  def __init__(self, bias, **kwargs):
    self.bias = bias
    super(ConstantBias, self).__init__(trainable=False, **kwargs)

  def build(self, input_shape):
    super(ConstantBias, self).build(input_shape)

  def call(self, x):
    y = x + self.bias
    return y

  def compute_output_shape(self, inputs_shape):
    return inputs_shape


class UseDTW(Layer):

  def __init__(self, **kwargs):
    super(UseDTW, self).__init__(trainable=False, **kwargs)

  def build(self, input_shape):
    super(UseDTW, self).build(input_shape)

  def call(self, inputs):
    #return x[self.dtw_y]
    x, dtw_y = inputs
    y = K.gather(x, dtw_y)
    return y

  def compute_output_shape(self, inputs_shape):
    return (inputs_shape[1][0], inputs_shape[0][1])


def compute_D2(a, b, La, Lb):
  #print((a, b, La, Lb))
  dtype = a.dtype
  #print(dtype)
  bT = K.transpose(b)
  sumaa = K.sum(a * a, axis=1, keepdims=True)
  sumbb = K.sum(b * b, axis=1, keepdims=True)
  #foo = K.ones((1, Lb), dtype=dtype)
  #D2 = K.dot(sumaa, foo)
  D2 = K.tile(sumaa, (1, Lb))
  D2 = D2 - 2 * K.dot(a, bT)
  #D2 = D2 + K.dot(K.ones((La, 1), dtype=dtype), K.transpose(sumbb))
  D2 = D2 + K.tile(K.transpose(sumbb), (La, 1))
  #D2 = K.dot(sumaa, K.ones((1, Lb), dtype=dtype)) - 2 * K.dot(a, bT) + K.dot(K.ones((La, 1), dtype=dtype), K.transpose(sumbb))
  D2 = K.abs(D2)
  return D2


def makeM0(La, Lb, thr, dtype):
  #am = K.arange(La) / (La - 1)
  am = K.arange(0, La, 1) / (La - 1)
  am = K.reshape(am, (La, 1))
  #bm = K.arange(Lb) / (Lb - 1)
  bm = K.arange(0, Lb, 1) / (Lb - 1)
  bm = K.reshape(bm, (Lb, 1))
  am = K.cast(am, dtype)
  bm = K.cast(bm, dtype)
  D2 = compute_D2(am, bm, La, Lb)
  M0 = 0.5 * 1e+10 * (1.0 + K.sign(D2 - thr * thr))
  return M0


def normList(l):
  sum = 0.0
  for i in l:
    sum = sum + i
  return [i / sum for i in l]


def compute_mask(La, Lb, dtype):
  am = K.arange(La) / (La - 1)
  am = K.reshape(am, (La, 1))
  bm = K.arange(Lb) / (Lb - 1)
  bm = K.reshape(bm, (Lb, 1))
  am = K.cast(am, dtype)
  bm = K.cast(bm, dtype)
  #M = 1.0 - compute_D2(am, bm, La, Lb)
  D2 = compute_D2(am, bm, La, Lb)
  M = K.exp(-4 * D2)
  if not M.dtype == dtype:
    M = K.cast(M, dtype)
  #threshold = 0.15
  #threshold2 = threshold * threshold
  #M = 0.5 + 0.5 * K.sign(threshold2 - D2)
  return M


def lossAttMSE(y_true, y_pred):
  a = y_pred
  b = y_true
  
  aShape = K.shape(a)
  La = aShape[-2]
  dima = aShape[-1]
  dtype = a.dtype
  a = K.reshape(a, (La, dima))

  bShape = K.shape(b)
  Lb = bShape[-2]
  dimb = bShape[-1]
  b = K.reshape(b, (Lb, dimb))
  
  D2_MSE = compute_D2(a, b, La, Lb)

  shifts =  [-3, -2, -1, 0, +1, +2, +3]
  wshifts = normList([0.125, 0.25, 0.5, 1.0, 0.5, 0.25, 0.125])
  a = applyTimeShifts(a, shifts, wshifts)
  b = applyTimeShifts(b, shifts, wshifts)

  if not a.dtype == dtype:
    a = K.cast(a, dtype)
    b = K.cast(b, dtype)
    
  M0 = makeM0(La, Lb, 0.1, dtype)
  D2 = compute_D2(a, b, La, Lb)
  mD2M = -(1.0 * D2 + M0)
  M1 = K.softmax(mD2M)
  M2 = K.transpose(K.softmax(K.transpose(mD2M)))
  M = M1 + M2

  return K.sum(K.sum(D2_MSE * M)) / K.sum(K.sum(M))


def MSE(y_true, y_pred):
  a = y_pred
  b = y_true

  # dealing with 3d arrays  
  aShape = K.shape(a)
  La = aShape[-2]
  dima = aShape[-1]
  dtype = a.dtype
  a = K.reshape(a, (La, dima))
  bShape = K.shape(b)
  Lb = bShape[-2]
  dimb = bShape[-1]
  b = K.reshape(b, (Lb, dimb))

  # dealing with different dtypes
  if not a.dtype == dtype:
    a = K.cast(a, dtype)
    b = K.cast(b, dtype)
  
  delta = a - b
  delta2 = delta * delta
  return K.mean(K.mean(delta2))
  
  
def compute_errorMSEDTW(y, tar, threshold=0.2):  
  d, path_y, path_tar = dtw(y, tar, threshold)
  delta = y[path_y] - tar[path_tar] 
  sum = numpy.sum(numpy.sum(delta * delta))
  #return sum, len(path_y), y.shape[1]
  return sum, delta.shape[0], delta.shape[1] 


def reshape_2d_fixed_dim(x, dim):
  xShape = K.shape(x)
  return K.reshape(x, (xShape[-2], dim))


def reshape_3d(x):
  xShape = K.shape(x)
  return K.reshape(x, (1, xShape[-2], xShape[-1]))
