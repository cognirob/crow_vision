from geometry_msgs.msg import Vector3, Quaternion
import numpy as np
import transforms3d as tf3


def make_vector3(tvec, order="xyz"):
    return Vector3(**{c: tt for c, tt in zip(map(str, order), np.ravel(tvec).astype(np.float32).tolist())})

def make_quaternion(quat, order="xyzw"):
    return Quaternion(**{c: tt for c, tt in zip(map(str, order), np.ravel(quat).astype(np.float32).tolist())})

def getTransformFromTF(tf_msg):
    trans = np.r_["0,2,0", [getattr(tf_msg.transform.translation, a) for a in "xyz"]]
    quat = [getattr(tf_msg.transform.rotation, a) for a in "xyzw"]
    rot = tf3.quaternions.quat2mat(quat[3:] + quat[:3])
    return trans, rot
