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


def computeDist(a, b):
  La = a.shape[0]
  Lb = b.shape[0]
  dim = a.shape[1]
  sumaa = numpy.sum(a * a, axis=1, keepdims=True)
  sumbb = numpy.sum(b * b, axis=1, keepdims=True)
  D2 = numpy.dot(sumaa, numpy.ones((1, Lb), dtype=a.dtype)) - 2 * numpy.dot(a, b.T) + numpy.dot(numpy.ones((La, 1), dtype=a.dtype), sumbb.T)
  return D2


def loadMatrixFloatText(fname, dtype="float32"):
  M = []
  f = open(fname)
  for line in f:
    M.append([float(x) for x in line.strip().split()])
  f.close()
  M = numpy.asarray(M, dtype=dtype)
  return M


if __name__ == "__main__":
  fnameIn = "data/keypoints-filt-deltas-norm.h5"
  mu = loadMatrixFloatText("temp/mu", dtype="float32")
  hfIn = h5py.File(fnameIn, "r")
  f = open("data/clear.list", "w")
  for key in hfIn:
    x = numpy.array(hfIn.get(key))
    delta = x - mu
    d = numpy.mean(numpy.sum(delta * delta, axis=1))
    print(d)
    if d < 150.0:
      f.write("%s\n" % key)
  f.close()
  hfIn.close()
