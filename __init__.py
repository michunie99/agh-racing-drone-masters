from gymnasium.envs.registration import register

register(
    id='race-aviary-v0',
    entry_point='envs:RaceEnv',
)