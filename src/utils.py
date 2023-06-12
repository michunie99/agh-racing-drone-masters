from pathlib import Path
import csv
from typing import Union
from collections import deque

import pybullet as p
import numpy as np


def calculateRelativeObseration(obj1, obj2):
    """
    obj = ([x, y, z], [qx, qy, qz, qw])
    """
    p1, quat1 = obj1
    p2, quat2 = obj2

    # TODO - keep info about quaterions and not shperical

    # Step 1 - transform points to obj1 cooridinate system
    # https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    vec = p2 - p1
    qvec = np.array([vec[0], vec[1], vec[2], 0])
    quat1_inv = inversQuaterion(quat1)
    qrot_vec = quaternion_multiply(quaternion_multiply(quat1,qvec),quat1_inv)
    rot_vec = qrot_vec[:-1]

    # Step 2 - calculate shperical coordinates
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    r = np.sqrt(np.sum(rot_vec**2))
    theta = np.arccos(rot_vec[2] / r)
    phi = np.sign(rot_vec[1]) * \
            np.arccos(rot_vec[0]/np.sqrt(np.sum(rot_vec[:1]**2)))
    
    # Step 3 - calculate angle between normals
    qobj_norm = np.array([0, 1, 0, 0])
    quat2_inv = inversQuaterion(quat2)
    obj_norm_cast = quaternion_multiply(quaternion_multiply(quat2_inv,qobj_norm),quat2)
    obj_norm = obj_norm_cast[:-1]
    vec_norm = vec / np.linalg.norm(vec)
    alpha = np.arccos(np.dot(vec_norm, obj_norm))

    return [r, theta, phi], alpha
    
def inversQuaterion(quat):
    quat = -1 * quat
    quat[-1] = -quat[-1]
    return quat

def quaternion_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)

if __name__ == "__main__":
    obj1 = (np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
    obj2 = (np.array([0, 1, 0]), np.array([0.0, 0.0, 0.3826834323650898, 0.9238795325112867]))

    print(calculateRelativeObseration(obj1, obj2))