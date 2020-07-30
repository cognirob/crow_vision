import pyrealsense2 as rs
import numpy as np
import cv2


class Diamond():

    def __init__(self, diamondId, markerIds=[]):
        self.__diamondId = diamondId
        self.__corners = None
        if len(markerIds) != 4:
            self.__markerIds = diamondId * 4 + np.r_[0, 1, 2, 3]
        else:
            self.__markerIds = markerIds

    def pushDetection(self, corners):
        self.__corners = corners

    @property
    def diamondId(self):
        return self.__diamondId

    @property
    def markerIds(self):
        return self.__markerIds

    @property
    def isFound(self):
        # TODO: include timestamp - if corners are stale, say isFound = False
        return self.__corners is not None

    @property
    def corners(self):
        return self.__corners


class FlatDiamondStack():

    def __init__(self, num_diamonds):
        if type(num_diamonds) is list:
            self.diamonds = {i: Diamond(i) for i in num_diamonds}
            self.num_diamonds = len(self.diamonds)
        else:
            self.num_diamonds = num_diamonds
            self.diamonds = {i: Diamond(i) for i in range(num_diamonds)}

    def checkDiamonds(self):
        fc = []
        for d in self.diamonds.values():
            if d.isFound:
                fc.append(d)
        return fc

    # def checkDiamonds(self):
    #     fc = 0
    #     for d in self.diamonds.values():
    #         if d.isFound:
    #             fc += 1
    #     return fc >= 3

    def pushDetection(self, diamondId, diamondCorners):
        if type(diamondId) is np.ndarray:
            diamondId = int(diamondId.ravel()[0] // 4)
        self.diamonds[diamondId].pushDetection(diamondCorners)
