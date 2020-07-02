%x = h5read('../data/demo/keypoints/keypoints-01-raw.h5','/20191005v')';
x = h5read('../data/demo/keypoints/keypoints-02-filter.h5','/20191005v')';

skeletonDisp(x)
