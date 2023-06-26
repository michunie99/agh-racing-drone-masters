import sys
import os
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym

sys.path.append(r'gym_pybullet_drones')
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from envs.RaceEnv import RaceAviary
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import EvalCallback

from gymnasium.envs.registration import register
from envs.enums import ScoreType


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
WORLD_BOX_SIZE = [5, 5, 3]
TRACK_PATH="tracks/single_gate.csv"
FILED_COEF=0.07
OMEGA_COEF=0.0005
GATE_FIELD_RANGE=-1
COMPLETION_TYPE=ScoreType.PLANE

register(
    id='race-aviary-v0',
    entry_point=lambda: RaceAviary( drone_model=DEFAULT_DRONES,
                                    initial_xyzs=np.array([0, 0, 1]).reshape(1, 3),
                                    initial_rpys=None,
                                    physics=DEFAULT_PHYSICS,
                                    freq=DEFAULT_SIMULATION_FREQ_HZ,
                                    gui=DEFAULT_GUI,
                                    record=DEFAULT_RECORD_VISION,
                                    gates_lookup=GATES_LOOKUP,
                                    score_radius=SCORE_RADIUS,
                                    world_box_size=WORLD_BOX_SIZE,
                                    track_path=TRACK_PATH,
                                    user_debug_gui=False, 
                                    filed_coef=FILED_COEF,
                                    omega_coef=OMEGA_COEF,
                                    completion_type=COMPLETION_TYPE,
                                    gate_filed_range=GATE_FIELD_RANGE
                                    ),
)

vec_env = DummyVecEnv([lambda: gym.make("race-aviary-v0")])
print("[INFO] Action space:", vec_env.action_space)
print("[INFO] Observation space:", vec_env.observation_space)

log_dir = "./logs/"# Use deterministic actions for evaluation

stats_path = os.path.join(log_dir, "vec_normalize.pkl")
VecNormalize.load(stats_path, vec_env)


model = PPO.load(log_dir + "ppo_race")
obs = vec_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    input()
    vec_env.render("human")
