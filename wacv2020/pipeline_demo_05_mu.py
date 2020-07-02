# Computing an average pose

import re
import os
import shutil
import codecs
import math
import copy
import h5py

import numpy

from utils import *


def saveMatrix(fname, x):
  f = open(fname, "w")
  for t in range(x.shape[0]):
    for i in range(x.shape[1]):
      f.write("%e\t" % x[t, i])
    f.write("\n")
  f.close()


if __name__ == "__main__":
  list_training = loadList("data/demo/lists/train.list")
  hfIn = h5py.File("data/demo/keypoints/keypoints-04-deltas-norm.h5", "r")
  sum0 = 0
  sum1 = 0.0
  sum2 = 0.0
  for key in hfIn:
    print("... reading %s" % key)
    x = numpy.array(hfIn.get(key))
    sum0 = sum0 + x.shape[0]
    sum1 = sum1 + numpy.sum(x, axis = 0, keepdims = True)
    sum2 = sum2 + numpy.sum(x * x, axis = 0, keepdims = True)
  hfIn.close()
  m = sum1 / sum0
  m2 = sum2 / sum0
  s = numpy.sqrt(m2 - m * m) 
  saveMatrix("temp/demo/mu", m)
