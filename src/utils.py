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
    # TODO - keep info about quaterions and not shperical
    pos1, ort1 = obj1
    pos2, ort2 = obj2

    # Step 1 - Vector between two points
    vec_diff = pos2 - pos1
    quat_diff = p.getDifferenceQuaternion(ort1, ort2)
    inv_p, inv_o = p.invertTransform([0,0,0], ort1)
    rot_vec, _ = p.multiplyTransforms(inv_p, inv_o,
                               vec_diff, [0, 0, 0, 1])

    # Step 2 - calculate shperical coordinates
    r, theta, phi = cart2shp(rot_vec)
    
    # Step 3 - calculate angle between normals
    _, alpha = p.getAxisAngleFromQuaternion(quat_diff)

    return np.array([r, theta, phi, alpha]).astype('float32')

def cart2shp(cart):
    xy = cart[0]**2 + cart[1]**2
    r = np.sqrt(xy + cart[2]**2)
    theta = np.arctan2(np.sqrt(xy), cart[2]) # for elevation angle defined from Z-axis down
    # sph[1] = np.arctan2(cart[2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    phi = np.arctan2(cart[1], cart[0])
    return r, theta, phi


if __name__ == "__main__":
    # Apply transfor and rotation to arg1 by arg2
    obj1 = np.array([0,0,0]), np.array([0.0,0,0.0,1])
    obj2 = np.array([1,1,1]), np.array([0.0, 0.0, 0.7071067811865475, 0.7071067811865476])

    pos1, ort1 = obj1
    pos2, ort2 = obj2

    # Vector between two points
    vec_diff = pos2 - pos1
    quat_diff = p.getDifferenceQuaternion(ort1, ort2)

    # Angle
    inv_p, inv_o = p.invertTransform([0,0,0], ort1)
    print(p.multiplyTransforms(inv_p, inv_o,
                               vec_diff, [0, 0, 0, 1]))
    print(p.getAxisAngleFromQuaternion(quat_diff))

    calculateRelativeObseration()
    # print(vec_diff, quat_diff)
    # print(p.multiplyTransforms(pos1, ort1,
    #                            vec_diff, quat_diff))
    
    # print(vec_diff)
    # print(p.getEulerFromQuaternion(quat_diff))
    # r = np.sqrt(np.sum(vec_diff**2))
    # print(p.multiplyTransforms([1,1,1], [0.0,0,0.0,1],
    #                            [1,1,1], [0.0, 0.0, 0.7071067811865475, 0.7071067811865476]))
    
    # print(p.invertTransform([1,1,1], [0.0,0,0.0,1]))