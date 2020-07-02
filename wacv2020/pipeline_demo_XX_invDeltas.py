# Converting from vectors (deltas) to absolute positions 

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


def computeInvDeltas(x):
  
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
  
  ndeltas = len(deltas)
  T, dim = x.shape
  n = int(dim / 2)

  X3 = numpy.zeros((T, 3 * (n + 1)), "float32")
  for t in range(T):

    for idelta in range(ndeltas):
      a, b = deltas[idelta]
  
      X3[t, 3 * b + 0] = X3[t, 3 * a + 0] - x[t, 2 * idelta + 0]
      X3[t, 3 * b + 1] = X3[t, 3 * a + 1] - x[t, 2 * idelta + 1]
  
  return X3


def saveAllData(fnameIn, fnameOut):
  hfIn = h5py.File(fnameIn, "r")
  hfOut = h5py.File(fnameOut, "w")
  for key in hfIn.keys():
    print(key)
    rec = numpy.array(hfIn.get(key))
    deltas = computeInvDeltas(rec)
    hfOut.create_dataset(key, data=deltas, dtype=deltas.dtype)    
  hfOut.close()
  hfIn.close()


if __name__ == "__main__":
  saveAllData(
    "...h5",
    "...-invDeltas.h5"
  )
