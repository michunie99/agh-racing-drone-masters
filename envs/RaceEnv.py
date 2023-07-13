import sys
import time
from collections import deque
from itertools import cycle

import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import pkg_resources


sys.path.append(r'gym_pybullet_drones')
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from src.track_loader import TrackLoader
from src.utils import calculateRelativeObseration, ProgresPath
from  .enums import ScoreType

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
                 gates_lookup=0,
                 score_radius=0.1,
                 normalize_state=False,
                 track_path="tracks/sample_track.csv",
                 world_box_size=[5, 5, 3],
                 omega_coef=0.001,
                 filed_coef=1,
                 completion_type=ScoreType.PLANE,
                 gate_filed_range=-1.5,
                 pos_off=None,
                 ort_off=None,
                 floor=True,
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
        self.score_radius=score_radius
        self.normlize_state=normalize_state

        self.track_path=track_path
        self.track_loader=TrackLoader(self.track_path)
        self.NUMBER_GATES=len(self.track_loader)
        self.obs_size=18 + (self.gates_lookup+1)*4
        self.current_gate=None
        self.prev_vel=0
        self.world_box_size=np.array(world_box_size)
        
        self.progress_tracker=ProgresPath()
        self.track_progress = [False for _ in range(self.NUMBER_GATES)]

        self.FILED_COEF=filed_coef
        self.OMEGA_COEF=omega_coef
        self.DMAX=-abs(gate_filed_range)
        
        self.completion_type=completion_type
        self.floor=floor
        
        if pos_off:
            self.POS_OFF=np.array(pos_off)
        else:
            self.POS_OFF=np.array([0, 0, 0])
            
        if ort_off:
            self.ORT_OFF=np.array(ort_off)
        else:
            self.ORT_OFF=np.array([0, 0, 0])
        
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
                         output_folder=output_folder,
                         )
        
        


    ################################################################################
    def _addObstacles(self):
        obs, ids = self.track_loader.loadTrack(self.CLIENT)
        self.GATE_IDS = ids
        self.OBS = obs
        
        self.gate_sizes=self.track_loader.gateWidth

        # if self.GUI and self.USER_DEBUG:
        #     for i in self.GATE_IDS:
        #         self._showDroneLocalAxes(i) #TODO: fix 
        # Add current gate to be gate 0
        self.current_gate_idx = 0

        # Update track progress tracker
        start_gate = p.getBasePositionAndOrientation(self.GATE_IDS[self.current_gate_idx], 
                                                    self.CLIENT)
        # state = self._getDroneStateVector(0)
        # drone_pos = state[0:3]

        drone_pos = self.INIT_XYZS[0,:]
        self.progress_tracker.updatePoints(drone_pos, start_gate[0])
        
        

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
        # act_lower_bound = np.array([0.,           0.,           0.,           0.])
        # act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        act_lower_bound = np.array([-1.,-1.,-1.,-1.])
        act_upper_bound = np.array([1., 1., 1., 1.])
        return spaces.Box(  low=act_lower_bound,
                            high=act_upper_bound,
                            dtype=np.float64)
    
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs. """
        # clipped_action = (np.array(action) + 1) / 2 * self.MAX_RPM #TODO fix this
            #     elif self.ACT_TYPE == ActionType.RPM:
            # return np.array(self.HOVER_RPM * (1+0.05*action))
        # Convert actions from [-1, 1] to [0, RPM_MAX]
        action = self._normalizedActionToRPM(action)
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
                            dtype=np.float64
        )
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment. """
        state = self._getDroneStateVector(0)

        if self.normlize_state:
            state = self._clipAndNormalizeState(state)
        
        # Calculate obseraction between drone and gate
        # obj = ([x, y, z], [qx, qy, qz, qw])
        drone_obj = state[0:3], state[3:7]
        gate_obj = p.getBasePositionAndOrientation(self.GATE_IDS[self.current_gate_idx], 
                                                   self.CLIENT)
        drone2gate = calculateRelativeObseration(drone_obj, gate_obj)
        gates = [drone2gate, *self.OBS[:self.gates_lookup]] #TODO - OBS not a cyclic structure

        rot_matrix = p.getMatrixFromQuaternion(state[3:7])
        acc = self._calculateAcceleration()

        obs = np.hstack([state[10:13], 
                         acc, 
                         rot_matrix, 
                         state[13:16], 
                         *gates]).reshape(self.obs_size,)
        return obs.astype(np.float64)


    ################################################################################
    def _calculateAcceleration(self):
        state = self._getDroneStateVector(0)
        vel = state[10:13]
        # print(self.prev_vel[0])
        acc = (vel - self.prev_vel) * self.SIM_FREQ
        self.prev_vel = vel
        # print(acc)
        return acc
    
    ################################################################################
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range. """
        #TODO: add notmaliaztion from sampled obseravations
        return state

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        int

        """
        reward = 0

        # Gate scored
        scored = self._gateScored(self.score_radius)
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        
        # Filed coeeficient
        wg = self.gate_sizes[self.current_gate_idx]

        d_pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        g_pos, g_ort = p.getBasePositionAndOrientation(self.GATE_IDS[self.current_gate_idx])
        
        # diff_vec = np.array(d_pos) - np.array(g_pos)
        diff_vec = np.array(g_pos) - np.array(d_pos)
        # Transform drone position to gate reference frame
        t_pos, _ = p.invertTransform(position=diff_vec,
                                     orientation=g_ort)
        
        # print(t_pos)
        dp, dn = t_pos[0], np.sqrt(t_pos[1]**2 + t_pos[2]**2)
        
        # print(f"dn: {dn:0.4f}, dp: {dp:0.4f}")
                
        f = lambda x: max(1-x/self.DMAX, 0.0)
        v = lambda x, y: max((1- y) * (x/6.0), 0.05)

        filed_reward = -f(dp)**2 * (1 - np.exp(-0.5 * dn**2 / v(wg, f(dp))))
        # print(filed_reward)
        
        reward += filed_reward * self.FILED_COEF
        

        # Add penaulty for gate colision
        # if len(p.getContactPoints(bodyA=self.DRONE_IDS[0],
                                #   bodyB=self.GATE_IDS[self.current_gate_idx],
                                #   physicsClientId=self.CLIENT)) != 0:
            # reward -= min((dn/wg)**2, 20)
        
        dg = np.linalg.norm(diff_vec) # Reward for crash as in paper
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0],
                                  physicsClientId=self.CLIENT)) != 0:
            reward -= min((dg/wg)**2, 20)
            
        # Add reguralization for the angular speed
        omega = state[13:16]
        omega_norm = np.linalg.norm(omega)**2
        # print(omega_norm)
        reward -= self.OMEGA_COEF * omega_norm
        progress = self.progress_tracker.calculateProgres(drone_pos)
            # print(progress)
        reward += progress 
              
        if scored:
            reward += 100
            self.current_gate_idx = self.current_gate_idx + 1
            
            if self.current_gate_idx == self.NUMBER_GATES:
                return reward

            # Update gate tracker
            start_gate = p.getBasePositionAndOrientation(self.GATE_IDS[self.current_gate_idx], 
                                                    self.CLIENT)
            self.progress_tracker.updatePoints(drone_pos, start_gate[0])

        return reward
    
    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s)."""

        if self.current_gate_idx == self.NUMBER_GATES:
            return True
        
        return False
           
    ################################################################################
    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        # print(np.abs(pos) > self.world_box_size)
        # Flew too far
        if np.any(np.abs(pos) > self.world_box_size):

            return True
        
        # Termiante when detected collision
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0],
                                 physicsClientId=self.CLIENT)) != 0:
            return True

        return False
    ################################################################################
    
    def _gateScored(self, thr):
        """ Compute if flew throught a gate """
        scored=False
        
        if self.completion_type==ScoreType.SHERE:
            state = self._getDroneStateVector(0)
            drone_obj = state[0:3], state[3:7]
            gate_obj = p.getBasePositionAndOrientation(self.GATE_IDS[self.current_gate_idx], 
                                                    self.CLIENT)
            r, _, _, _ = calculateRelativeObseration(drone_obj, gate_obj)

            if r < thr:
                scored = True
        
        elif self.completion_type==ScoreType.PLANE:
            d_pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
            g_pos, g_ort = p.getBasePositionAndOrientation(self.GATE_IDS[self.current_gate_idx])
            
            state = self._getDroneStateVector(0)
            vel = state[10:13]
        
            # diff_vec = np.array(d_pos) - np.array(g_pos)
            diff_vec = np.array(g_pos) - np.array(d_pos)
            # Transform drone position to gate reference frame
            t_pos, _ = p.invertTransform(position=diff_vec,
                                        orientation=g_ort)
            
            # Transform velocity
            t_vel, _ = p.invertTransform(position=vel,
                                        orientation=g_ort)
            wg = self.gate_sizes[self.current_gate_idx]
            
            x, y, z = t_pos[0], t_pos[1], t_pos[2]
            
            if (0.0 < abs(x) < 0.01 
                and abs(y) < wg/2 
                and abs(z) < wg/2
                and t_vel[0] <= 0):
                scored = True
        
        return scored

    ################################################################################

    def _housekeeping(self):
        super()._housekeeping()
        
        # My house keeping
        self.current_gate_idx = 0
        self.prev_vel = np.array([0, 0, 0])
        
        if not self.floor:
            p.removeBody(self.PLANE_ID, self.CLIENT)
        
        # # TODO: add initial state randomization if flag specified
        # for i in self.DRONE_IDS:
        #     d_pos, d_ort = p.getBasePositionAndOrientation(i)
        #     pos_off = self.POS_OFF * np.random.normal(  loc=0.0, 
        #                                                 scale=1.0, 
        #                                                 size=(3, ),
        #                                                 )
            
        #     ort_off = self.ORT_OFF * np.random.normal(  loc=0.0, 
        #                                                 scale=1.0, 
        #                                                 size=(3, ),
        #                                                 )
        #     d_pos += pos_off
        #     _, quat_off = p.getQuaternionFromEuler(ort_off)
        #     d_ort = self._quaternionMultiply( d_ort, 
        #                                         quat_off,
        #                                      )
        #     p.resetBasePositionAndOrientation(i, 
        #                                       posObj=d_pos,
        #                                       ornObj=d_ort,
        #                                       physicsClientId=self.CLIENT,
        #                                       )
    ################################################################################
    
    
    def _quaternionMultiply(quaternion1, quaternion0):
        """Return multiplication of two quaternions.

            >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
            >>> numpy.allclose(q, [28, -44, -14, 48])
            True

            """
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0, x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0, x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ],
                            dtype=np.float64)
     
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
