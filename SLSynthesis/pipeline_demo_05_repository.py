# -*- coding: utf8 -*-

import re
import os
import shutil
import codecs
import math
import copy
import h5py
import random

import numpy

from utils import *


def saveMatrix(fname, x):
  f = open(fname, "w")
  for t in range(x.shape[0]):
    for i in range(x.shape[1]):
      f.write("%e\t" % x[t, i])
    f.write("\n")
  f.close()


def computeDist(a, b):
  La = a.shape[0]
  Lb = b.shape[0]
  dim = a.shape[1]
  sumaa = numpy.sum(a * a, axis=1, keepdims=True)
  sumbb = numpy.sum(b * b, axis=1, keepdims=True)
  D2 = numpy.dot(sumaa, numpy.ones((1, Lb), dtype=a.dtype)) - 2 * numpy.dot(a, b.T) + numpy.dot(numpy.ones((La, 1), dtype=a.dtype), sumbb.T)
  return D2


def loadMatrixFloatText(fname, dtype="float64"):
  M = []
  f = open(fname)
  for line in f:
    M.append([float(x) for x in line.strip().split()])
  f.close()
  M = numpy.asarray(M, dtype=dtype)
  return M


if __name__ == "__main__":
  K = 1024
  
  fnameIn = "data/keypoints-filt-deltas-norm.h5"
  fnameRep = "temp/repository-%d" % K
  
  hfIn = h5py.File(fnameIn, "r")

  iters = 20

  rep = numpy.zeros((K, 2 * 49), dtype="float32")
  
  keys = loadList("data/training.list")
  clear = loadList("data/clear.list")
  keys = [key for key in keys if key in clear]
  
  if True:
    print("0")
    sum0 = numpy.zeros((K, ), dtype="int64")
    sum1 = numpy.zeros(rep.shape, dtype="float32")
    i = 0
    for key in keys:
      x = numpy.array(hfIn.get(key))
      for t in range(0, x.shape[0], 7):
        x_t = x[t]
        #i = random.randint(0, K - 1)
        sum0[i] = sum0[i] + 1
        sum1[i] = sum1[i] + x_t
        i = (i + 1) % K
        if i == 0:
          break
      if i == 0:
        break
      #break
    rep = sum1 / (sum0.reshape((K, 1)) + 1e-30)
    saveMatrix(fnameRep, rep)  
  else:
    rep = loadMatrixFloatText(fnameRep, dtype="float32")

  for iter in range(iters):
    sume = 0
    sume0 = 0
    print(iter + 1)
    sum0 = numpy.zeros((K, ), dtype="int64") + 1
    sum1 = numpy.zeros(rep.shape, dtype="float32") + rep
    for key in keys:
      x = numpy.array(hfIn.get(key))
      d = computeDist(x, rep)
      dm = numpy.min(d, axis=1)
      dargm = numpy.argmin(d, axis=1)
      cands = [(dm[i], dargm[i], i) for i in range(dm.shape[0])]

      for k in range(len(cands)):
        m, am, t = cands[k]
        sum0[am] = sum0[am] + 1
        sum1[am] = sum1[am] + x[t]
        sume = sume + m
        sume0 = sume0 + 1

    rep = (sum1 / (sum0.reshape((K, 1)) + 1e-30))
    print(sum0)
    print(float(sume) / float(sume0))
  
    saveMatrix(fnameRep, rep)  

  hfIn.close()
