import sys
import time

import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import pkg_resources


sys.path.append(r'gym_pybullet_drones')
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from src.track_loader import TrackLoader

class RaceAviary(BaseAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 user_debug_gui=True,
                 output_folder='results',
                 gates_lookup=1,
                 normalize_state=False,
                 track_path="tracks/sample_track.csv"
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        self.gates_lookup=gates_lookup
        self.normlize_state=normalize_state

        self.track_path=track_path
        self.track_loader=TrackLoader(self.track_path)
        self.NUMBER_GATES=len(self.track_loader)
        self.obs_size=19 + (self.NUMBER_GATES-1)*4

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=True,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )
        
        


    ################################################################################
    def _addObstacles(self):
        obs, ids = self.track_loader.loadTrack(self.CLIENT)
        self.GATE_IDS = ids
        self.OBS = obs

    ################################################################################
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([0.,           0.,           0.,           0.])
        act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Box(  low=act_lower_bound,
                            high=act_upper_bound,
                            dtype=np.float32)
    
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs. """
        #TODO - netwrok output to RPM
        return action

    ################################################################################
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Box of drone observations
        """
        #### Observation vector ### 
        ###                            Vx       Vy        Vz      Ax        Ay       Az      r1*9,    wx       wy      wz              gates*n
        # obs_lower_bound = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, *[-1]*9, -np.inf, -np.inf, -np.inf, [0, -2*np.pi, -np.pi]*self.gates_lookup])
        # obs_upper_bound = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, *[1]*9, np.inf, np.inf, np.inf, *[np.inf, 2*np.pi, np.pi]*self.gates_lookup])
        # return spaces.Box( low=obs_lower_bound,
        #                     high=obs_upper_bound,
        #                     dtype=np.float32
        # )             
        return spaces.Box( low=np.ones((self.obs_size, ))*-np.inf,
                            high=np.ones((self.obs_size, ))*np.inf,
                            dtype=np.float32
        )
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment. """
        state = self._getDroneStateVector(0)

        if self.normlize_state:
            state = self._clipAndNormalizeState(state)

        rot_matrix = p.getMatrixFromQuaternion(state[3:7])
        # gates = self.gates_queue[:self.gates_lookup]
        gates = self.OBS
        acc = self._calculateAcceleration()
        # TODO fix obervations !!!
        obs = np.hstack([state[9:13], acc, rot_matrix, state[12:15], gates[0]]).reshape(self.obs_size,)
        return obs.astype('float32')


    ################################################################################
    def _calculateAcceleration(self):
        #TODO
        return np.array([0, 0, 0])
    
    ################################################################################
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range. """
        #TODO
        return state

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s)."""
        return False
        if len(p.getContactPoints(self.DRONE_IDS[0], self.CLIENT)):
            return True
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
