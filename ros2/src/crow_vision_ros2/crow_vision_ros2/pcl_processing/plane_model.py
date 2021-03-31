import numpy as np
import open3d as o3d


class PlaneModel():

    def __init__(self, distance_threshold=0.01, ransac_n=3, iterations=1000):
        self.__samples = np.zeros((0, 3), dtype=np.float64)
        self.__distance_threshold = distance_threshold
        self.__ransac_n = ransac_n
        self.__iterations = iterations

        self.__normal = np.ones((3))
        self.__offset = np.ones((1))

    def addSamples(self, samples):
        self.__samples = np.vstack((self.__samples, np.array(samples)))

    def addSamplesOpen3d(self, samples):
        self.addSamples(samples.points)

    def compute(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.__samples)
        plane_model, _ = pcd.segment_plane(distance_threshold=self.__distance_threshold,
                                           ransac_n=self.__ransac_n,
                                           num_iterations=self.__iterations)
        self.__normal = plane_model[:-1].astype(np.float32)
        self.__offset = plane_model[-1].astype(np.float32)

    def filter(self, points, removeAllBelow=True):
        return points[self.argFilter(points, removeAllBelow)]

    def filterOpen3d(self, pcl, removeAllBelow=True):
        points = np.array(pcl.points, dtype=np.float32)
        return self.filter(points, removeAllBelow)

    def argFilter(self, points, removeAllBelow=True):
        if removeAllBelow:
            return np.where((np.dot(points, self.__normal) + self.__offset) < self.__distance_threshold)
        else:
            return np.where(np.abs(np.dot(points, self.__normal) + self.__offset) > self.__distance_threshold)

    def argFilterPlaneOnly(self, points, uptoHeight=0.1, removeAllBelow=True):
        distances = np.dot(points, self.__normal) + self.__offset
        if removeAllBelow:
            abovePlane_idxs = distances < self.__distance_threshold
        else:
            abovePlane_idxs = np.abs(distances) > self.__distance_threshold
        return np.where(np.logical_and(abovePlane_idxs, np.abs(distances) < uptoHeight))

    def argFilterOpen3d(self, pcl, removeAllBelow=True):
        points = np.array(pcl.points, dtype=np.float32)
        return self.argFilter(points, removeAllBelow)

    @property
    def normal(self):
        return self.__normal

    @property
    def offset(self):
        return self.__offset

    @property
    def n_samples(self):
        return self.__samples.shape[0]
