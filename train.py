import argparse
import time
from typing import Union
import pickle
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

import wandb
from wandb.integration.sb3 import WandbCallback

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.build_network import DronePolicy
from envs import RaceAviary
from envs.enums import ScoreType

wandb.login()
    
def parse_args():
    parser = argparse.ArgumentParser()
    
    # Env parameters
    parser.add_argument("--norm-reward", action="store_true",
                        help="Normalize reward in vector enviroment")
    parser.add_argument("--norm-obs", action="store_true",
                        help="Normalize observation in vector enviroment")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Number of time steps to run trainign")
    parser.add_argument("--num-valid", type=int, default=10,
                        help="Number of validation enviroments")
    
    # TODO - add validation runs
    parser.add_argument("--time-valid", type=int, default=1_000,
                        help="Number of steps between validation steps")
 
    parser.add_argument("--model-save-freq", type=int, default=1_000,
                        help="Frequeny of model save")
    parser.add_argument("--num-env", type=int, default=6,
                        help="Number of parallel envirements to")
    parser.add_argument("--track-path", type=str, default='tracks/single_gate.csv',
                            help="Path for the track file")
    parser.add_argument("--cpus", type=int, default=1, 
                        help="Number of CPU cores, determines number of vec envs") 
    parser.add_argument("--gate-lookup", type=int, default=0,
                        help="Gate lookup in the observation space")
    parser.add_argument("--world-box", type=list, default=[5, 5, 3],
                        help="Size of the world box")
    parser.add_argument("--field-coef", type=float, default=0.0005,
                        help="Gate filed coefficient in the reward")
    parser.add_argument("--omega-coef", type=float, default=0.00001,
                        help="Angular velocity coefficient in the reward")
    parser.add_argument("--compl-type", type=ScoreType, default=ScoreType.PLANE,
                        help="Way of determening if gate was passed succesfully")
    parser.add_argument("--gate-field-range", type=float, default=-1.5,
                        help="Gate filed range in [m]")
    parser.add_argument("--dn-scale", type=float, default=0.7,
                        help="Scaling factor for the data serach")
    parser.add_argument("--ort-off", type=tuple, default=(0, 0.1),
                        help="Offset for the orientation")
    parser.add_argument("--pos-off", type=tuple, default=(0, 0),
                        help="Offset for the possition")
    
    # PPO parameters
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3,
                        help="Learining rate for PPO")
    parser.add_argument("--gamma", "-g", type=float, default=0.97,
                        help="Discount factor in MDP")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used for experiment")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="path to models dir")
    parser.add_argument("--run-number", type=int, default=None,
                        help="number of run to run")
    
    
    # wb and logging
    parser.add_argument("--wb-logging", "-wb", action="store_true",
                        help="Weather to log experiment to wb")
    parser.add_argument("--logs-dir", "-l", type=str, default="./logs",
                        help="Dirpath to save experimnet logs") 
    
    # Gui
    parser.add_argument("--gui", action="store_true",
                        help="Display gui for debugging")
    
    args = parser.parse_args()
    return args
    
def make_env(args, gui):
    env_builder = lambda: gym.make(
        "race-aviary-v0",
        drone_model=DroneModel('cf2x'),
        initial_xyzs=np.array([0, 0, 1]).reshape(1, 3), # TODO - add to parser
        # TODO - change the intialil_xyzs
        initial_rpys=None,
        physics=Physics('pyb'),
        freq=240,
        gui=gui,
        record=False,
        gates_lookup=args.gate_lookup,
        world_box_size=args.world_box,
        track_path=args.track_path,
        user_debug_gui=False, 
        filed_coef=args.field_coef,
        omega_coef=args.omega_coef,
        completion_type=args.compl_type,
        gate_filed_range=args.gate_field_range,
        dn_scale=args.dn_scale,
        ort_off=args.ort_off,
        pos_off=args.pos_off,
    )
    return env_builder

def run(args):
    # Create enviroments
        
    envs = [make_env(args, args.gui) for _ in range(args.cpus)]
    vec_env = SubprocVecEnv(envs)
        
    if args.model_dir and args.run_number:
        model_dir = Path(args.model_dir)
        model_path = model_dir / f"{model_dir.stem}_race_model_{args.run_number}_steps.zip"
        norm_path = model_dir / f"{model_dir.stem}_race_model_vecnormalize_{args.run_number}_steps.pkl"
        
        vec_env = VecNormalize.load(
                norm_path,
                vec_env
        )
    else:
        # Create a normalization wrapper
        vec_env = VecNormalize(
            vec_env,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_reward,
            # TODO - add clipping in config
        )

    # Add monitor wrapper
    vec_env = VecMonitor(
        vec_env,
        filename=args.logs_dir
    ) 

    # Set up wdb
    curr_time = time.gmtime()
    run_id = f'{time.strftime("%d_%m_%y_%H_%M", curr_time)}_race_exp'
    
    # Create logs file
    logs_path = Path(f"./logs/models/{run_id}")
    if not logs_path.exists():
        logs_path.mkdir(parents=True)

    # Save args
    with open(f"./logs/models/{run_id}/args.pkl", "bw") as f:
        pickle.dump(args, f)
    
    callbacks = []
    if args.wb_logging: 
        run = wandb.init(
            project="drone_race",
            config=args,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
        )
        callbacks.append(WandbCallback(
            gradient_save_freq=1000,
            #model_save_path=f'models/{run.id}',
            verbose=2,
            # TODO - what else to add ???
        ))
        
    callbacks.append(CheckpointCallback(
            save_freq=args.model_save_freq,
            save_path=f"./logs/models/{run_id}",
            name_prefix=f"{run_id}_race_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
    ))
    
    if not args.model_dir:
        model = PPO(
            DronePolicy,
            vec_env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            seed=args.seed,
            verbose=1,
            tensorboard_log="./logs/tensor_board/", #TODO - run id
        )
    else:
        model = PPO.load(
                model_path,
                vec_env,
      )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=run_id,
    )
    # TODO Add model save !!!

    if args.wb_logging: 
        run.finish()
    
if __name__ == "__main__":
    args = parse_args()

    run(args) 
