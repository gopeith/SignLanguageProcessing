import h5py
import re
import random
import os
import codecs
import sys
from datetime import datetime 

import numpy


def loadText(fname, cod="utf8"):
  f = codecs.open(fname, "r", cod)
  text = {}
  ws = {}
  for line in f:
    foo = line.strip().split()
    key = foo[0]
    sent = foo[1:len(foo)]
    for w in sent:
      if not w in ws:
        ws[w] = len(ws)
    sent_int = [ws[w] for w in sent]
    text[key] = numpy.asarray(sent_int, dtype="int32")
  f.close()
  return text, len(ws)
  

def loadInfo(fname, dtype, cod="utf8"):
  f = codecs.open(fname, "r", cod)
  info = {}
  vals = {}
  for line in f:
    key, val = line.strip().split()
    if not val in vals:
      vals[val] = len(vals)
    info[key] = val
  f.close()
  for key in info.keys():
    val = info[key]
    info[key] = numpy.zeros((1, len(vals), ), dtype=dtype)
    info[key][0, vals[val]] = 1
  return info, len(vals)


def loadDenses(fname, names):
  result = []
  hf = h5py.File(fname, "r")
  for name in names:
    #print(name)
    gr = hf.get(name)[name]
    w = gr["kernel:0"]
    w = numpy.array(w)
    result.append(w)
    if "bias:0" in gr:
      b = gr["bias:0"]
      b = numpy.array(b)
      result.append(b)
    #else:
    #  b = numpy.zeros((w.shape[1], ), dtype=w.dtype)
  hf.close()
  return result


def loadAllDenses(fname):
  result = []
  hf = h5py.File(fname, "r")
  names = []
  i = 1
  while "dense_%d" % i in hf:
    names.append("dense_%d" % i)
    i = i + 1
  for name in names:
    gr = hf.get(name)[name]
    w = gr["kernel:0"]
    w = numpy.array(w)
    result.append(w)
    if "bias:0" in gr:
      b = gr["bias:0"]
      b = numpy.array(b)
      result.append(b)
    #else:
    #  b = numpy.zeros((w.shape[1], ), dtype=w.dtype)
  hf.close()
  return result


def loadAllDenses2(fname, maxNum=None):
  result = []
  hf = h5py.File(fname, "r")
  names = []
  if maxNum is None:
    i = 1
    while "dense2_%d" % i in hf:
      names.append("dense2_%d" % i)
      i = i + 1
  else:
    for i in range(1, maxNum + 1):
      if "dense2_%d" % i in hf:
        names.append("dense2_%d" % i)
  for name in names:
    gr = hf.get(name)[name]
    w = gr["kernel:0"]
    w = numpy.array(w)
    result.append(w)
    if "bias:0" in gr:
      b = gr["bias:0"]
      b = numpy.array(b)
      result.append(b)
    #else:
    #  b = numpy.zeros((w.shape[1], ), dtype=w.dtype)
  hf.close()
  return result


def loadList(fname):
  f = open(fname)
  l = [line.strip() for line in f]
  f.close()
  return l


def saveMatrixText(fname, x):
  f = open(fname, "w")
  for t in range(x.shape[0]):
    for i in range(x.shape[1]):
      f.write("%e\t" % x[t, i])
    f.write("\n")
  f.close()


def loadMatrixFloatText(fname, dtype="float64"):
  M = []
  f = open(fname)
  for line in f:
    M.append([float(x) for x in line.strip().split()])
  f.close()
  M = numpy.asarray(M, dtype=dtype)
  return M

