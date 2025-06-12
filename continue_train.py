#!/usr/bin/env python
import argparse
from stable_baselines3 import PPO
from Env import Walker2DEnv

def continue_training(model_path, reset_timesteps):
    # 환경 생성
    env = Walker2DEnv("xml/2D_Quad_SPINE.xml")
    # 이전에 저장한 모델 로드 (env를 같이 인자로 넣어야 합니다)
    model = PPO.load(model_path, env=env)
    # 이어서 학습 (reset_num_timesteps=False 하면 이전 timestep을 유지)
    total_timesteps = 3000000
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=reset_timesteps)
    # 이어서 학습한 모델 저장
    model.save("ppo_walker2d_continued_model")
    print("Continued model saved as ppo_walker2d_continued_model.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continue training a PPO model on Walker2DEnv."
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="If set, do NOT reset num_timesteps (continue training from previous timestep)."
    )
    parser.add_argument(
        "-path",
        type=str,
        default="models/PPO_9900000_steps.zip",
        help="Path to the saved model."
    )
    args = parser.parse_args()

    # --continue-training가 있으면 reset_num_timesteps=False
    reset_num_timesteps = not args.continue_training
    continue_training(args.path, reset_num_timesteps)


# 이어서 학습:  python continue_train.py -path models/PPO_1000000_steps.zip