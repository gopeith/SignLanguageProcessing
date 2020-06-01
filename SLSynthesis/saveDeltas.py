# -*- coding: utf8 -*-

import re
import os
import shutil
import codecs
import math
import copy
import h5py

import numpy


dtypeFloat = "float32"
dtypeInt = "int64"


def saveMatrix(fname, x):
  f = open(fname, "w")
  for t in range(x.shape[0]):
    for i in range(x.shape[1]):
      f.write("%e\t" % x[t, i])
    f.write("\n")
  f.close()


def computeDeltas(x):
  
  deltas = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),

    (4, 29),
    
    (29, 30),
    (30, 31),
    (31, 32),
    (32, 33),

    (29, 34),
    (34, 35),
    (35, 36),
    (36, 37),

    (29, 38),
    (38, 39),
    (39, 40),
    (40, 41),

    (29, 42),
    (42, 43),
    (43, 44),
    (44, 45),

    (29, 46),
    (46, 47),
    (47, 48),
    (48, 49),

    (7, 8),

    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (8, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (8, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (8, 21),
    (21, 22),
    (22, 23),
    (23, 24),

    (8, 25),
    (25, 26),
    (26, 27),
    (27, 28),
  )
  
  T = x.shape[0]
  n = x.shape[1] / 3
  nd = len(deltas)
  y = numpy.zeros((T, 2 * nd), dtype=dtypeFloat)
  for t in range(T):
    for iDelta in range(nd):
      i = deltas[iDelta][0]
      j = deltas[iDelta][1]
      y[t, 2 * iDelta + 0] = x[t, 3 * i + 0] - x[t, 3 * j + 0]
      y[t, 2 * iDelta + 1] = x[t, 3 * i + 1] - x[t, 3 * j + 1]
  return y 


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


#def saveAllData():
#  data = {}
#  for fnameInput in walkDir("../../data/keypoints/txt", r"\.txt$"):
#    print fnameInput
#    fnameClear = re.sub(r".*/", "", re.sub(r"\.[^\.]+$", "", fnameInput))
#    rec = loadRec(fnameInput)
#    rec = numpy.asarray(rec, dtype=dtypeFloat)
#    deltas = computeDeltas(rec)
#    saveMatrix("../../data/keypoints/deltas/" + fnameClear + ".txt", deltas)


def saveAllData(fnameIn, fnameOut):
  hfIn = h5py.File(fnameIn, "r")
  hfOut = h5py.File(fnameOut, "w")
  for key in hfIn.keys():
    print(key)
    rec = numpy.array(hfIn.get(key))
    deltas = computeDeltas(rec)
    hfOut.create_dataset(key, data=deltas, dtype=deltas.dtype)    
  hfOut.close()
  hfIn.close()


if __name__ == "__main__":
  #saveAllData()
  saveAllData("../../data/keypoints/keypoints-clean-skeleton.h5", "../../data/keypoints/keypoints-clean-skeleton-deltas.h5")
