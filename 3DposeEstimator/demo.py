# standard
import h5py

# 3rd party
import numpy

# our own 
import skeletalModel
import pose2D
import pose2Dto3D
import pose3D 


def save(fname, lst):

  T, dim = lst[0].shape
  f = open(fname, "w")
  for t in range(T):
    for i in range(dim):
      for j in range(len(lst)):
        f.write("%e\t" % lst[j][t, i])
    f.write("\n")
  f.close()


if __name__ == "__main__":
  
  dtype = "float32"
  randomNubersGenerator = numpy.random.RandomState(1234)

  # This demo shows converting a result of 2D pose estimation into a 3D pose.
  
  # Getting our structure of skeletal model.
  # For customizing the structure see a definition of getSkeletalModelStructure.  
  structure = skeletalModel.getSkeletalModelStructure()
  
  # Getting 2D data
  # The sequence is an N-tuple of
  #   (1sf point - x, 1st point - y, 1st point - likelihood, 2nd point - x, ...)
  # a missing point should have x=0, y=0, likelihood=0 
  f = h5py.File("data/demo-sequence.h5", "r")
  inputSequence_2D = numpy.array(f.get("20161025_pocasi"))
  f.close()
  
  # Decomposition of the single matrix into three matrices: x, y, w (=likelihood)
  X = inputSequence_2D
  Xx = X[0:X.shape[0], 0:(X.shape[1]):3]
  Xy = X[0:X.shape[0], 1:(X.shape[1]):3]
  Xw = X[0:X.shape[0], 2:(X.shape[1]):3]
  
  # Normalization of the picture (x and y axis has the same scale)
  Xx, Xy = pose2D.normalization(Xx, Xy)
  save("data/demo1.txt", [Xx, Xy, Xw])

  # Delete all skeletal models which have a lot of missing parts.
  Xx, Xy, Xw = pose2D.prune(Xx, Xy, Xw, (0, 1, 2, 3, 4, 5, 6, 7), 0.3, dtype)
  save("data/demo2.txt", [Xx, Xy, Xw])
  
  # Preliminary filtering: weighted linear interpolation of missing points.
  Xx, Xy, Xw = pose2D.interpolation(Xx, Xy, Xw, 0.99, dtype)
  save("data/demo3.txt", [Xx, Xy, Xw])
  
  # Initial 3D pose estimation
  lines0, rootsx0, rootsy0, rootsz0, anglesx0, anglesy0, anglesz0, Yx0, Yy0, Yz0 = pose2Dto3D.initialization(
    Xx,
    Xy,
    Xw,
    structure,
    0.001, # weight for adding noise
    randomNubersGenerator,
    dtype
  )
  save("data/demo4.txt", [Yx0, Yy0, Yz0])
    
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
  )
  save("data/demo5.txt", [Yx, Yy, Yz])

  
