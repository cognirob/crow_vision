import numpy as np
from numpy.random import random
import torch
from torch import tensor
from time import time
from scipy.stats import mode
from scipy.spatial import cKDTree
from uuid import uuid4
from numba import jit


class Position():

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Orientation():

    @classmethod
    def fromQuaternion(cls, *, quat=None, x=0.0, y=0.0, z=0.0, w=0.0):
        return cls()  # TODO: somehow convert quat to rotation matrix or something

    def __init__(self, sxyz=[0.0, 0.0, 0.0]):
        """Create orientation representation object.
        Should hold the orientation of an object in matrix and some other

        Parameters
        ----------
        sxyz : list, optional
            Euler angles, by default [0.0, 0.0, 0.0]
        """
        pass


class ObjectModel():

    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class Timer():

    def __init__(self):
        self._timer = time
        self._lastTime = self._timer()

    def getTimeDiff(self):
        now = self._timer()
        diff = now - self._lastTime
        self._lastTime = now
        return diff

    @property
    def now(self):
        return self._timer()

    @property
    def lastTime(self):
        return self._lastTime


object_properties = {
    0: {
        "name": "car_roof",
        "sigma": 0.2  # TODO
    },
    1: {
        "name": "cube_holes",
        "sigma": 0.05
    },
    2: {
        "name": "ex_bucket",
        "sigma": 0.1  # TODO
    },
    3: {
        "name": "hammer",
        "sigma": 0.0  # TODO
    },
    4: {
        "name": "nut",
        "sigma": 0.025
    },
    5: {
        "name": "peg_screw",
        "sigma": 0.06
    },
    6: {
        "name": "peg_simple",
        "sigma": 0.06
    },
    7: {
        "name": "pliers",
        "sigma": 0.2  # TODO
    },
    8: {
        "name": "screw_round",
        "sigma": 0.13
    },
    9: {
        "name": "screwdriver",
        "sigma": 0.2  # TODO
    },
    10: {
        "name": "sphere_holes",
        "sigma": 0.05  # TODO
    },
    11: {
        "name": "wafer",
        "sigma": 0.05  # TODO
    },
    12: {
        "name": "wheel",
        "sigma": 0.075
    },
    13: {
        "name": "wrench",
        "sigma": 0.2  # TODO
    }
}


class ParticleFilter():
    MEAN_SHIFT_PRECISSION = 0.001
    CLASS_HISTORY_LEN = 10
    PARTICLES_PER_MODEL = 1000
    TIMER_CLASS = Timer
    DELETION_TIME_LIMIT = 10  # 10 seconds
    MAX_SHIFT_ITERS = 30
    STATIC_NOISE = 0.05  # 5 cm
    GLOBAL_DISTANCE_LIMIT = 0.3  # limit for a model to be considered "close" to new observation
    TREE_DISTANCE_UPPER_BOUND = 0.01  # TODO: should be probably 1mm?
    CLOSE_MODEL_PROBABILITY_THRESHOLD = 1e-2  # TODO: search for optimal value
    PARTICLES_PCL_COUNT = 500  # how many particles to generate from measured PCLs
    PARTICLES_UNIFORM_COUNT = 100  # how many uniformly distributed particles to generate
    PARTICLES_UNIFORM_DISTANCE = 0.5  # size of circle or cube around the object in which the uniformly random ptcs should be generated
    MODEL_SHIFT_NOISE_LIMIT = 0.005  # if model moves less than this, it is considered a noise, not an actual movement
    ACCELERATION_LIMIT = 0.5  # 1m/s**2 is assumed as max acceleration (higher values are clipped)
    SPEED_LIMIT = 2  # 1m/s is assumed as max speed (higher values are clipped)

    def __init__(self):
        self.timer = self.TIMER_CLASS()
        self.model_particles = None  # tensor QxMx3
        self.particle_weights = None  # tensor QxMx1
        self.model_params = None  # tensor Qx2x3  (3-velocity & 3-acceleration)
        self.model_states = None  # tensor Qx3  (estimated position)
        self.model_classes = None  # ndarray Qx1
        self.model_class_history = None  # ndarray QxH
        self.model_class_names = None  # ndarray Qx1
        self.model_last_update = None  # ndarray Qx1
        self.model_uuid = None  # ndarray of str Q,
        self.n_models = 0
        self.observations = []
        self.__uniform_noise_generator = torch.distributions.uniform.Uniform(-self.PARTICLES_UNIFORM_DISTANCE, self.PARTICLES_UNIFORM_DISTANCE)
        self.__uniform_noise_size = torch.Size((self.PARTICLES_UNIFORM_COUNT, 3))

    @property
    def classes(self):
        return self.model_classes

    @property
    def estimates(self):
        return self.model_states.numpy()

    def update(self):
        if self.n_models == 0:
            if len(self.observations) > 0:
                self._processMeasurements()
            return  # nothing to update

        time_delta = self.timer.getTimeDiff()
        time_now = self.timer.lastTime

        # remove stale models
        toBeDeleted = np.where((self.model_last_update - time_now) > self.DELETION_TIME_LIMIT)[0].tolist()
        if len(toBeDeleted) > 0:
            for tbd in toBeDeleted:
                self._delete_model(tbd)
        # move
        self._predict(time_delta)
        # diffuse
        self.model_particles.add_(torch.randn_like(self.model_particles) * self.STATIC_NOISE * time_delta)

        # add measurements
        if len(self.observations) > 0:
            self._processMeasurements()
            # estimate classes
            self._estimate_classes()  # no need to estimate classes when no new measurements arrived

        # estimate
        self._estimate(time_delta)

    def getEstimates(self):
        """Return a tuple of "xyz" position and class name (with uuid)
        """
        return [(xyz.numpy(), name + "_" + id[:id.find("-")]) for xyz, name, id in zip(self.model_states, self.model_class_names, self.model_uuid)]

    def add_measurement(self, z):
        self.observations.append(z)

    def _predict(self, time_delta):
        """
        * move models according to current state
        * spread particles
        * add uniformly random particles
        """
        time_tensor = tensor([time_delta, time_delta**2], dtype=torch.float32).unsqueeze(0)  # (1x2)
        self._update_velocities = time_tensor.matmul(self.model_params)  # model_params : (Qx2x3) -> Qx1x3
        # TODO: update velocity and acceleration?
        self.model_particles.add_(self._update_velocities)  # model_particles : Qx1000x3

    def _estimate(self, time_delta):
        # save previous states
        old_states = self.model_states.clone()
        # merge particles and weights
        samples_weights = torch.cat((self.model_particles, self.particle_weights.unsqueeze(-1)), dim=-1).split(1)
        # update position estimates from particles
        self.model_states = torch.tensor([self._estimate_model(sw.squeeze().numpy()) for sw in samples_weights])
        # estimate model parameters (speed & acceleration) TODO: do Kalman filtering?
        # TODO: add gradient backwards error propagation
        model_shifts = self.model_states.sub(old_states)  # could be considered as current actual speed
        shift_magnitudes = torch.norm(model_shifts, dim=1)

        not_moved_idx = shift_magnitudes < self.MODEL_SHIFT_NOISE_LIMIT
        previously_stationary = self.model_params[:, 0, :].sum(dim=1) == 0
        velocity = model_shifts.div(time_delta)
        acceleration = model_shifts.sub(self._update_velocities.squeeze(1)).div(time_delta**2)

        self.model_params = torch.cat((
            velocity.unsqueeze(1),
            acceleration.unsqueeze(1)
        ), dim=1)
        self.model_params[not_moved_idx, :] = torch.zeros((2, 3))
        self.model_params[previously_stationary, 1, :] = torch.zeros((1, 3))
        self.model_params[:, 0, :].clamp_(-self.SPEED_LIMIT, self.SPEED_LIMIT)  # TODO: limit on speed magnitude
        self.model_params[:, 0, :].clamp_(-self.ACCELERATION_LIMIT, self.ACCELERATION_LIMIT)
        """
        a = (speed_now - speed_last) / time_delta
        """

    def _estimate_classes(self):
        # update classes
        cls_estimate = mode(self.model_class_history, axis=1, nan_policy="propagate")[0]
        cls_estimate[np.isnan(cls_estimate)] = -1
        self.model_classes = cls_estimate.ravel()
        self.model_class_names = [object_properties[class_key]["name"] if class_key in object_properties else "unknown" for class_key in self.model_classes]

    def _processMeasurements(self):
        """
        For each observed PCL:
            * check if close to any model -> assign to model (check probability first) or create new model

        Parameters
        ----------
        observations : list of PCL
            List of point clouds to be processed.
        """
        # helper variables
        self._model_pcls = np.empty(self.n_models, dtype=np.object)
        self._model_trees = np.empty(self.n_models, dtype=np.object)

        # go through every observation
        for pcl, label in self.observations:
            assert pcl.shape[1] == 3
            if self.n_models == 0:  # no models exists -> automatically create model for each PCL
                self._add_model(pcl, label)
                continue
            # compute PCL center
            pcl_center = np.median(pcl, axis=0).reshape(1, 3)
            # compute distances
            distances2models = torch.norm(self.model_states.sub(tensor(pcl_center, dtype=torch.float32)), dim=1)
            mclose = distances2models < self.GLOBAL_DISTANCE_LIMIT  # TODO: th based on model sigma
            mclose_idx = torch.where(mclose)[0].tolist()
            if len(mclose_idx) > 0:
                close_models = np.zeros(self.n_models)
                # append pcl to closest model
                for idx in mclose_idx:
                    cm_tree = self._model_trees[idx]
                    if cm_tree is None:  # tree not created yet, create new tree
                        close_model = self.model_particles[idx, ...]
                        cm_tree = cKDTree(close_model)
                        self._model_trees[idx] = cm_tree  # cache the model
                    # query tree for 1 neighbors
                    p_dist, p_idx = cm_tree.query(pcl, k=1, eps=1e-4, distance_upper_bound=self.TREE_DISTANCE_UPPER_BOUND)
                    valid = p_dist < np.inf
                    if not np.any(valid):  # no particle is close to points in the PCL
                        continue
                    p_idx = p_idx[valid]  # take points with pairs
                    close_models[idx] = self.particle_weights[idx, p_idx].sum()  # sum weights
                # calculate model with highest probability
                if np.any(close_models) and np.max(close_models) > self.CLOSE_MODEL_PROBABILITY_THRESHOLD:
                    closest_model = np.argmax(close_models)
                    self._append_pcl_to_model(closest_model, pcl, label, pcl_center)
                else:
                    self._add_model(pcl, label, pcl_center)
            else:  # no model is close, add new model
                self._add_model(pcl, label, pcl_center)

        # go through aggregated PCLs for each model
        for idx, mpcls in enumerate(self._model_pcls):
            if mpcls is None:
                continue
            pcls = np.empty((0, 3), dtype=np.float)
            pcl_centers = np.empty((0, 3), dtype=np.float)
            labels = []
            for pcl, pcl_center, label in mpcls:
                pcls = np.vstack((pcls, pcl))
                pcl_centers = np.vstack((pcl_centers, pcl_center))
                labels.append(label)
            # aggregate labels from PCLs
            self.model_class_history[idx, :] = np.roll(self.model_class_history[idx, :], -1, axis=0)  # shift old data out
            self.model_class_history[idx, -1] = mode(labels)[0][0]
            # create particles form PCLs (downsample PCL)
            particles_pcl = tensor(pcls[np.random.choice(np.arange(pcls.shape[0]), self.PARTICLES_PCL_COUNT), :], dtype=torch.float32)
            # create uniformly random particles
            particles_uniform = self._generate_uniform_noise(self.model_states[idx, ...])
            # aggregate particles
            particles = torch.cat((self.model_particles[idx, ...], particles_pcl, particles_uniform))
            # compute PCLs mean
            center = np.median(pcl_centers, axis=0)
            # sigma from model (add sigma from pcl? <- np.std(pcls, axis=0))
            sigma = object_properties[self.model_classes[idx]]["sigma"]
            # resample
            resamp_particles, resamp_weights = self._weigh_resample_particles(particles, center, sigma, self.PARTICLES_PER_MODEL)
            self.model_particles[idx, ...] = resamp_particles  # resampled particles are of the same type as input, so no need for "tensorification"
            self.particle_weights[idx, ...] = tensor(resamp_weights, dtype=torch.float32)

        # clear observations
        self.observations = []

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _systematic_resample(weights, n_samples):
        positions = (np.arange(n_samples) + random()) / n_samples

        indexes = np.zeros(n_samples).astype(np.int32)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < n_samples:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    @staticmethod
    def _weigh_resample_particles(particles, mean, sigma, n_particles):
        if type(particles) == torch.Tensor:
            points = particles.numpy()
        else:
            points = particles

        # weight by gaussian distribution
        weights = (np.exp(-((points - mean))**2 / (2 * sigma**2)) / (sigma * np.sqrt(np.pi * 2))).sum(axis=1)
        # normalize weights
        weights /= weights.sum()
        # resample
        resamp_idx = ParticleFilter._systematic_resample(weights, n_particles)
        w_resampled = weights[resamp_idx]
        return particles[resamp_idx, :], w_resampled / w_resampled.sum()

    def _generate_uniform_noise(self, mean):
        return self.__uniform_noise_generator.sample(self.__uniform_noise_size).add_(mean)

    def _append_pcl_to_model(self, model_idx, pcl, est_class, pcl_center=None):
        if pcl_center is None:
            pcl_center = np.median(pcl, axis=0).reshape(1, 3)
        if self._model_pcls[model_idx] is None:
            self._model_pcls[model_idx] = [(pcl, pcl_center, est_class)]
        else:
            self._model_pcls[model_idx].append((pcl, pcl_center, est_class))

    def _add_model(self, pcl, class_est, pcl_center=None):
        if pcl_center is None:
            pcl_center = np.median(pcl, axis=0).reshape(1, 3)

        n_pcl = pcl.shape[0]

        sigma = object_properties[class_est]["sigma"]
        # generate particles
        noisy_pcl = pcl + np.random.randn(n_pcl, 3) * self.STATIC_NOISE
        m_particles, p_weights = self._weigh_resample_particles(noisy_pcl, pcl_center, sigma, self.PARTICLES_PER_MODEL)
        m_particles = tensor(m_particles, dtype=torch.float32).unsqueeze(0)
        p_weights = tensor(p_weights, dtype=torch.float32).unsqueeze(0)
        # m_particles = torch.zeros((1, self.PARTICLES_PER_MODEL, 3))
        # p_weights = torch.ones((1, self.PARTICLES_PER_MODEL, 1)).div_(self.PARTICLES_PER_MODEL)
        m_params = torch.zeros((1, 2, 3))
        m_state = tensor(pcl_center, dtype=torch.float32)
        m_class_history = np.ones((1, self.CLASS_HISTORY_LEN)) * np.nan
        # m_class_history[..., -1] = class_est  # this is actually done in processing model pcls, no need to do it here
        last_update = self.timer.now
        id = str(uuid4())
        class_name = [object_properties[class_est]["name"] if class_est in object_properties else "unknown"]
        update_velocity = torch.zeros((1, 1, 3))

        if self.n_models == 0:
            self.model_particles = m_particles
            self.particle_weights = p_weights
            self.model_params = m_params
            self.model_states = m_state
            self.model_classes = np.r_[class_est]
            self.model_class_history = m_class_history
            self.model_last_update = np.r_[last_update]
            self.model_uuid = np.array([id])
            self.model_class_names = class_name
            self._update_velocities = update_velocity
        else:
            self.model_particles = torch.cat((self.model_particles, m_particles))
            self.particle_weights = torch.cat((self.particle_weights, p_weights))
            self.model_params = torch.cat((self.model_params, m_params))
            self.model_states = torch.cat((self.model_states, m_state))
            self.model_classes = np.r_[self.model_classes, class_est]
            self.model_class_history = np.vstack((self.model_class_history, m_class_history))
            self.model_last_update = np.r_[self.model_last_update, last_update]
            self.model_uuid = np.hstack((self.model_uuid, id))
            self.model_class_names += class_name
            self._update_velocities = torch.cat((self._update_velocities, update_velocity))

        self._model_pcls = np.hstack((self._model_pcls, None))
        self._model_trees = np.hstack((self._model_trees, None))
        self._append_pcl_to_model(self.n_models, pcl, class_est, pcl_center)
        self.n_models += 1

    def _delete_model(self, model_number):
        mask = torch.arange(0, self.n_models) != model_number

        self.model_particles = self.model_particles[mask, ...]
        self.particle_weights = self.particle_weights[mask, ...]
        self.model_params = self.model_params[mask, ...]
        self.model_states = self.model_states[mask, ...]
        self.model_classes = self.model_classes[mask, ...]
        self.model_class_history = self.model_class_history[mask, ...]
        self.model_last_update = self.model_last_update[mask, ...]
        self.model_uuid = self.model_uuid[mask, ...]
        self.model_class_names = np.array(self.model_class_names)[torch.where(mask)[0].tolist()].tolist()
        self._update_velocities = self._update_velocities[mask, ...]

        self.n_models -= 1

    def _estimate_model(self, samples_weights):
        shift = np.inf
        samples = samples_weights[..., :3].squeeze()
        weights = samples_weights[..., 3].squeeze()
        mode = np.average(samples, axis=0, weights=weights)
        iters = 0
        try:
            while shift > self.MEAN_SHIFT_PRECISSION or iters < self.MAX_SHIFT_ITERS:
                mode_old = mode
                dist = np.linalg.norm(samples - mode_old, axis=1)
                th_dist = np.median(dist)
                sub_idx = dist < th_dist
                mode = np.average(samples[sub_idx, :], axis=0, weights=weights[sub_idx])
                shift = np.linalg.norm(mode - mode_old)
                iters += 1
        except Exception as e:  # noqa
            return np.average(samples, axis=0, weights=weights)
        else:
            return mode