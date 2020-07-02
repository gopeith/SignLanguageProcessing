# Normalizing standard deviance of all points (remember sigma if you want to transform it back to pixels)

import re
import os
import shutil
import codecs
import math
import copy
import h5py

import numpy


dtypeFloat = "float64"
dtypeInt = "int64"


def saveMatrix(fname, x):
  f = open(fname, "w")
  for t in range(x.shape[0]):
    for i in range(x.shape[1]):
      f.write("%e\t" % x[t, i])
    f.write("\n")
  f.close()


def walkDir(dname, filt = r".*"):
  result = []
  for root, dnames, fnames in os.walk(dname):
    for fname in fnames:
      if re.search(filt, fname):
        foo = root + "/" + fname
        foo = re.sub(r"[/\\]+", "/", foo)
        result.append(foo)
  return result


def loadRec(fname):
  rec = []
  f = open(fname)
  for line in f:
    x = []
    for x_i in line.strip().split():
      x.append(float(x_i))
    rec.append(x)
  f.close()
  return rec


def saveAllData(fnameIn, fnameOut):
  hfIn = h5py.File(fnameIn, "r")

  sum0 = 0
  sum1 = 0.0
  sum2 = 0.0
  for key in hfIn:
    #print(key)
    x = numpy.array(hfIn.get(key))
    sum0 = sum0 + x.shape[0]
    sum1 = sum1 + numpy.sum(x, axis = 0, keepdims = True)
    sum2 = sum2 + numpy.sum(x * x, axis = 0, keepdims = True)
  m = sum1 / sum0
  m2 = sum2 / sum0
  s = numpy.sqrt(m2 - m * m)
  sigma = numpy.mean(s) 
  
  print("sigma = %e" % sigma)
    
  hfOut = h5py.File(fnameOut, "w")
  for key in hfIn.keys():
    #print(key)
    rec = numpy.array(hfIn.get(key))
    rec = rec / sigma
    hfOut.create_dataset(key, data=rec, dtype=rec.dtype)    
  hfOut.close()
  hfIn.close()
    

if __name__ == "__main__":
  saveAllData(
    "data/demo/keypoints/keypoints-03-deltas.h5",
    "data/demo/keypoints/keypoints-04-deltas-norm.h5",
  )
