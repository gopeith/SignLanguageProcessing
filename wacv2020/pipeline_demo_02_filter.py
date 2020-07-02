# Using a filter to interpolate missing kezpoints.
 

# standard
import h5py

# 3rd party
import numpy

# our own 
import skeletalModel
import pose2D
import pose2Dto3D
import pose3D


def convList2Array(lst): 
  T, dim = lst[0].shape
  a = []
  for t in range(T):
    a_t = []
    for i in range(dim):
      for j in range(len(lst)):
        a_t.append(lst[j][t, i])
    a.append(a_t)
  return numpy.asarray(a)


def use_filter(x):
  structure = skeletalModel.getSkeletalModelStructure()

  inputSequence_2D = x
  
  # Decomposition of the single matrix into three matrices: x, y, w (=likelihood)
  X = inputSequence_2D
  Xx = X[0:X.shape[0], 0:(X.shape[1]):3]
  Xy = X[0:X.shape[0], 1:(X.shape[1]):3]
  Xw = X[0:X.shape[0], 2:(X.shape[1]):3]
  
  # Normalization of the picture (x and y axis has the same scale)
  Xx, Xy = pose2D.normalization(Xx, Xy)

  # Delete all skeletal models which have a lot of missing parts.
  Xx, Xy, Xw = pose2D.prune(Xx, Xy, Xw, (0, 1, 2, 3, 4, 5, 6, 7), 0.3, dtype)
  
  # Preliminary filtering: weighted linear interpolation of missing points.
  Xx, Xy, Xw = pose2D.interpolation(Xx, Xy, Xw, 0.99, dtype)
  
  # Initial 3D pose estimation
  lines0, rootsx0, rootsy0, rootsz0, anglesx0, anglesy0, anglesz0, Yx0, Yy0, Yz0 = pose2Dto3D.initialization(
    Xx,
    Xy,
    Xw,
    structure,
    0.001, # weight for adding noise
    randomNubersGenerator,
    dtype,
    percentil=0.5,
  )
    
  # Backpropagation-based filtering
  Yx, Yy, Yz = pose3D.backpropagationBasedFiltering(
    lines0, 
    rootsx0,
    rootsy0, 
    rootsz0,
    anglesx0,
    anglesy0,
    anglesz0,   
    Xx,   
    Xy,
    Xw,
    structure,
    dtype,
    nCycles=100,
  )
  
  return convList2Array([Yx, Yy, Yz])


if __name__ == "__main__":
  


  fnameIn = "data/demo/keypoints/keypoints-01-raw.h5"
  fnameOut = "data/demo/keypoints/keypoints-02-filter.h5"

  dtype = "float32"
  randomNubersGenerator = numpy.random.RandomState(1234)


  hfIn = h5py.File(fnameIn, "r")
  hfOut = h5py.File(fnameOut, "w")
  for key in hfIn:
    print("")
    print("... processing '%s'" % key)
    print("")
    x = numpy.array(hfIn.get(key))
    y = use_filter(x)
    hfOut.create_dataset(key, data=y, dtype=y.dtype)
  hfOut.close()
  hfIn.close()

