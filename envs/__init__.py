from gymnasium.envs.registration import register
from envs.RaceEnv import RaceAviary

register(
    id='race-aviary-v0',
    entry_point=RaceAviary
)