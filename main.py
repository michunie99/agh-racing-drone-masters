"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import sys
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

sys.path.append(r'gym_pybullet_drones')
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from envs.RaceEnv import RaceAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
GATES_LOOKUP = 0
SCORE_RADIUS = 0.2

env = RaceAviary(drone_model=DEFAULT_DRONES,
                    initial_xyzs=np.array([0, 0, 1]).reshape(1, 3),
                    initial_rpys=None,
                    physics=DEFAULT_PHYSICS,
                    freq=DEFAULT_SIMULATION_FREQ_HZ,
                    gui=DEFAULT_GUI,
                    record=DEFAULT_RECORD_VISION,
                    gates_lookup=GATES_LOOKUP,
                    score_radius=SCORE_RADIUS,
                    )

env.reset()
#### Obtain the PyBullet Client ID from the environment ####
PYB_CLIENT = env.getPyBulletClient()



p.setGravity(0.0, 0.0, 0.0, physicsClientId=PYB_CLIENT)
p.applyExternalForce(env.DRONE_IDS[0], -1, [0.8, 0, 0], [0, 0, 0], p.LINK_FRAME)

progress = []
i = 0


while True:
    obs, reward, terminated, truncated, info = env.step([0,0,0,0])
    
    if  i % 100 == 0:
        progress.append(env.progress_tracker.last_poss)
        # print(env.progress_tracker.last_poss)
    print(p.getContactPoints(env.DRONE_IDS[0], PYB_CLIENT))
    
    i += 1
    if terminated or truncated:
        break

    # print(obs)
    
    # env.render()

plt.figure()
plt.plot(progress)
plt.show()