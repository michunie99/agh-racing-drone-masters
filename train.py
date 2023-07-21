import argparse
import time
from typing import Union

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3 import PPO

import wandb
from wandb.integration.sb3 import WandbCallback

from src.build_network import DronePolicy

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Env parameters
    parser.add_argument("--norm-reward", action="store_true",
                        help="Normalize reward in vector enviroment")
    parser.add_argument("--norm-obs", action="store_true",
                        help="Normalize observation in vector enviroment")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000,
                        help="Number of time steps to run trainign")
    parser.add_argument("--num-valid", type=int, default=10,
                        help="Number of validation enviroments")
    parser.add_argument("--time-valid", type=int, default=1_000,
                        help="Number of steps between validation steps")
    parser.add_argument("--num-env", type=int, default=6,
                        help="Number of parallel envirements to")
    parser.add_argument("--track-dirpath", type=str,
                            help="Path for the track file")
    parser.add_argument("--cpus", type=int, 
                        help="Number of CPU cores, determines number of vec envs") 
    
    # PPO parameters
    parser.add_argument("--lerning-rate", "-lr", type=float, default=1e-3,
                        help="Learining rate for PPO")
    parser.add_argument("--gamma", "-g", type=float, default=0.99,
                        help="Discount factor in MDP")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used for experiment")
    parser.add_argument("--fine-tune", type=Union[str, None], default=None,
                        help="Use a pretrained model to fine tune")
    
    # wb and logging
    parser.add_argument("--wb-logging", "-wb", action="store_true",
                        help="Weather to log experiment to wb")
    parser.add_argument("--logs-dir", "-l", type=str, default="./logs",
                        help="Dirpath to save experimnet logs")
    
    
def make_env(args, gui):
    pass

def run(args):
    # Create enviroments
    envs = [make_env(args) for _ in range(args.cpus)]
    vec_env = SubprocVecEnv(envs)
     
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
   
    callbacks = []
    if args.wb_logging: 
        run = wandb.init(
            project="drone_race",
            config=args,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            run_id=run_id,
        )
        callbacks.append(WandbCallback(
            gradient_save_freq=100,
            model_save_path=f'models/{run.id}',
            verbose=2,
            # TODO - what else to add ???
        ))
        
    model = PPO(
        DronePolicy,
        vec_env,
        verbose=1,
        tensorboard_log="./logs/ppo_test_drone/", #TODO - run id
    )
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks
    )
    
    run.finish()
    
if __name__ == "__main__":
   args = parse_args()
   
   run(args) 