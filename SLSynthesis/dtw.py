from datetime import datetime 

import numpy


def min(l):
  m = None
  am = None
  for i in range(len(l)):
    if m is None or m > l[i]:
      m = l[i] 
      am = i
  return m, am


def min3(a, b, c):
  if a > b:
    if b > c:
      return c, 2
    return b, 1
  if a > c:
    return c, 2
  return a, 0


def isOut(pa, pb, La, Lb, t=0.1):
  qa = pa / float(La)
  qb = pb / float(Lb)
  return qa - qb > t or qb - qa > t


def dtw(a, b, maxDelta=1.0, D=None):
  dtype = a.dtype

  inf = 1e+30
  
  La = a.shape[0]
  Lb = b.shape[0]
  dim = a.shape[1]
  
  phi = inf + numpy.zeros((La, Lb), dtype=dtype)
  psi = -1 + numpy.zeros((La, Lb), dtype="int64")
  
  if D is None:
    # computing distance matrix  
    sumaa = numpy.sum(a * a, axis=1, keepdims=True)
    sumbb = numpy.sum(b * b, axis=1, keepdims=True)
    D2 = numpy.dot(sumaa, numpy.ones((1, Lb), dtype=dtype)) - 2 * numpy.dot(a, b.T) + numpy.dot(numpy.ones((La, 1), dtype=dtype), sumbb.T)
    D = D2
  
  LbLa = float(Lb) / float(La)
  
  for ia in range(La):
    #for ib in range(Lb):
    ibMin = int(LbLa * (ia + 1) - maxDelta * Lb - 1)
    ibMax = int(LbLa * (ia + 1) + maxDelta * Lb - 1)
    if ibMin < 0:
      ibMin = 0
    if ibMax > Lb - 1:
      ibMax = Lb - 1
    for ib in range(ibMin, ibMax + 1):
      if ia == 0 and ib == 0:
        phi[ia, ib] = D[ia, ib]
      else:
        #delta = ((ia + 1.0) / float(La)) - ((ib + 1.0) / float(Lb))
        #if delta > maxDelta or delta < -maxDelta:
        #  continue
        if ia - 1 >= 0 and ib - 1 >= 0:
          if phi[ia, ib] > phi[ia - 1, ib - 1]:
            phi[ia, ib] = phi[ia - 1, ib - 1]
            psi[ia, ib] = 3
        if ia - 1 >= 0:
          if phi[ia, ib] > phi[ia - 1, ib]:
            phi[ia, ib] = phi[ia - 1, ib]
            psi[ia, ib] = 1
        if ib - 1 >= 0:
          if phi[ia, ib] > phi[ia, ib - 1]:
            phi[ia, ib] = phi[ia, ib - 1]
            psi[ia, ib] = 2
        if phi[ia, ib] < inf:
          phi[ia, ib] = phi[ia, ib] + D[ia, ib]
      
  
  point = (La - 1, Lb - 1)
  patha = [point[0]]    
  pathb = [point[1]]
  while True:
    stepType = psi[point[0], point[1]]
    if stepType <= 0:
      break
    if stepType == 1:
      point = (point[0] - 1, point[1])
    if stepType == 2:
      point = (point[0], point[1] - 1)
    if stepType == 3:
      point = (point[0] - 1, point[1] - 1)
    patha.insert(0, point[0])
    pathb.insert(0, point[1])

  return phi[La - 1, Lb - 1], patha, pathb



def fakedtw(a, b, maxDelta="whatever"):
  if a.shape[0] > b.shape[0]:
    T = a.shape[0]
  else:
    T = b.shape[0]
  patha = numpy.zeros((T, ), dtype="int64")
  pathb = numpy.zeros((T, ), dtype="int64")
  for t in range(T):
    patha[t] = int(((a.shape[0] - 1) * t) / (T - 1))
    pathb[t] = int(((b.shape[0] - 1) * t) / (T - 1))
  return None, patha, pathb


