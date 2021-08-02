import numpy as np
from numpy.random import randn
from scipy.stats import norm as stat_norm
from filterpy.monte_carlo import stratified_resample


class ParticleFilter():

    def __init__(self, n_particles:int = 100, initial_guess = 5, spread_factor = 1, measurement_noise = 0.005, window_size = 30, window_max_var = 5):
        self.n_particles = n_particles
        self.n_dim = np.shape(initial_guess)[0]
        self.__initParticles(initial_guess, [spread_factor] * self.n_dim)
        self.neff_threshold = self.n_particles / 2
        self.spread_factor = spread_factor
        self.measurement_noise = measurement_noise

        self.window_size = window_size
        self.window = np.zeros(window_size)
        self.window_idx = 0
        self.window_max_var = window_max_var

    def step(self, measurement):
        self.addNoise()
        self.update(measurement)
        self.__mean = self.estimate()
        # self.__mean, self.__var = self.estimate()
        self.window[self.window_idx % self.window_size] = self.__mean
        self.window_idx += 1

        # print(f"Estimated position = {mean} (var = {var})")
        if self.neff() < self.neff_threshold:
            self.regenerate()

    def update(self, measurement):
        self.weights *= stat_norm(self.particles, self.measurement_noise).pdf(measurement) + 1.e-300
        self.weights /= self.weights.sum()

    def estimate(self):
        mean = np.average(self.particles, weights=self.weights, axis=0)
        # var  = np.average((self.particles - mean)**2, weights=self.weights, axis=0)
        return mean#, var

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def regenerate(self):
        self.particles = self.particles[stratified_resample(self.weights)]
        self.weights.fill(1 / self.n_particles)

    def __initParticles(self, mean, sigma):
        self.particles = np.empty((self.n_particles, self.n_dim))
        self.weights = np.ones((self.n_particles, 1)) / self.n_particles

        for i in range(self.n_dim):
            self.particles[:, i] = mean[i] + (randn(self.n_particles) * sigma[i])

    def addNoise(self):
        self.particles += randn(self.n_particles, self.n_dim) * self.spread_factor

    @property
    def mean(self):
        return self.__mean

    @property
    def smooth_mean(self):
        return np.median(self.window[np.abs(self.window - np.median(self.window)) < np.var(self.window) * self.window_max_var])


class MultiFilter():

    def __init__(self, n_points, n_particles_per_pt, noise, initial_guesses=None):
        if initial_guesses is None:
            initial_guesses = [0] * n_points
        self.filters = [ParticleFilter(n_particles_per_pt, initial_guess, noise * 10, noise) for initial_guess in initial_guesses]

    def step(self, measurements):
        for flt, meas in zip(self.filters, measurements):
            flt.step(meas)


class Marker2D():

    def __init__(self, id, length=5, max_deviation=1):
        self.__id = id
        self.__length = length
        self.__windowIdx = 0
        self.max_deviation = max_deviation
        self.__corner_window = np.zeros((self.length, 4, 2), dtype=np.float32)

    @property
    def length(self):
        return self.__length

    @property
    def id(self):
        return self.__id

    @property
    def corners(self):
        slc = np.all(np.reshape(np.abs(self.__corner_window - np.median(self.__corner_window, axis=0)) < np.std(self.__corner_window, axis=0) * self.max_deviation, (self.length, -1)), axis=1)
        if any(slc):
            return np.mean(self.__corner_window[slc, :], axis=0)
        else:
            return self.__corner_window[self.__windowIdx % self.length, :]

    @corners.setter
    def corners(self, corner_points):
        self.__corner_window[self.__windowIdx % self.__length] = np.squeeze(corner_points).reshape(4, 2)
        self.__windowIdx += 1

    @property
    def ready(self):
        return self.__windowIdx > self.length

    @property
    def window(self):
        return self.__corner_window


class CameraPoser():

    def __init__(self, name):
        self.name = name
        self.__markers = {}
        self.__diamonds = {}

    def updateMarker(self, id, corners):
        id = np.ravel(id).tolist()[0]
        if id not in self.__markers:
            self.__markers[id] = Marker2D(id)
        self.__markers[id].corners = corners

    def getMarkers(self):
        return [mrk.corners[np.newaxis] for mrk in self.__markers.values()], np.array([id for id in self.__markers.keys()], dtype=np.int32)[:, np.newaxis]

    def updateDiamond(self, id, corners):
        if type(id) is not str:
            id = str(id)
        if id not in self.__diamonds:
            self.__diamonds[id] = Marker2D(id)
        self.__diamonds[id].corners = corners

    @property
    def markersReady(self):
        return all([mrk.ready for mrk in self.__markers.values()])

    @property
    def markers(self):
        return self.__markers

    @property
    def diamonds(self):
        return self.__diamonds


class CameraFilter():

    def __init__(self):
        self.__cameras = {}


    def addCamera(self, cameraName):
        self.__cameras[cameraName] = CameraPoser(cameraName)

    @property
    def cameras(self):
        return self.__camera



class CameraParticleFilter():

    def __init__(self, n_cameras:int, n_particles_marker:int = 200, n_particles_diamond:int = 500, n_particles_point:int = 1000):
        """Initialize camera calibration filter

        Args:
            n_cameras (int): number of cameras or viewpoints existing in the scene
            n_particles_marker (int, optional): Number of filter particles per marker. Defaults to 200.
            n_particles_diamond (int, optional): Number of filter particles per ChAruCo diamond. Defaults to 500.
            n_particles_point (int, optional): Number of filter particles per object point (i.e. real world correspondence points for the diamond marker points). Defaults to 1000.
        """
        pass

    def updateMarkers(self, cameraId, markerIds, markerCorners):
        """Updates the positions of the markers per camera.

        Args:
            cameraId (any): Camera identifier.
            markerIds (int): Identification numbers of the detected markers.
            markerCorners (list): Corner locations of the individual detected markers.
        """
        pass

    def estimateMarkerPositions(self, cameraId):
        """Computes marker position estimates for all markers from the specified camera.
        """
        pass

    def updateDiamonds(self, cameraId, diamondIds, diamondCorners):
        """Similar to "updateMarkers"

        Args:
            cameraId (any): Camera identifier.
            diamondId (list of int): Diamond id.
            diamondCorners (list of lists): Diamond corners.
        """
        pass

    def estimateDiamondPositions(self, cameraId):
        """Computes diamond position estimates for all diamonds from the specified camera.
        """
        pass


    def __initParticles(self, n_particles, mean, sigma):
        pass

    def __addNoise(self, particles):
        pass
