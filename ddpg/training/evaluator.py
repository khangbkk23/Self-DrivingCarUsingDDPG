from sqlalchemy import exists
import torch
import numpy as np
import gymnasium as gym
import os
from gymnasium.wrappers import RecordVideo
from utils.preprocess import ImagePreProcessor

class Evaluator:
    def __init__(self, config, device):
        self.cfg = config
        self.device = device
        self.env_name = config['env_name']
        self.eval_env = gym.make(self.env_name, continuous=True)
        
        
    def _preprocess(self, state):
        state = state.transpose(2,0,1)
        state = torch.FloatTensor(state).unsqueeze(0).to(device=self.device)
        return state / 255.0
    
    def evaluate(self, actor, n_episodes=5):
        actor.eval()
        total_reward = 0.0
        
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            
            max_steps = self.cfg['training'].get('max_steps_per_episode', 1000)
            for _ in range(max_steps):
                with torch.no_grad():
                    state_input = self._preprocess(state)
                    action = actor(state_input).cpu().data.numpy().flatten()
                
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
            
        actor.train()
        avg_reward = total_reward / n_episodes
        return avg_reward
    
    def save_evaluation_video(self, actor, filename_prefix="best_model"):
        video_folder = "./videos/"
        os.makedirs(video_folder, exist_ok=True)
        env = gym.make(self.env_name, continuous=True, render_mode="rgb_array")

        env = RecordVideo(
            env, 
            video_folder, 
            name_prefix=filename_prefix,
            episode_trigger=lambda x: True,
            disable_logger=True
        )

        actor.eval()
        processor = ImagePreProcessor()
        state, _ = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                state_input = processor.process(state)

                state_tensor = torch.FloatTensor(state_input).unsqueeze(0).to(self.device)
                state_tensor = ImagePreProcessor.normalize(state_tensor) # / 255.0
                
                action = actor(state_tensor).cpu().data.numpy().flatten()
            
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        env.close()
        print(f"--> Video saved to {video_folder}{filename_prefix}-episode-0.mp4")
        
    def close(self):
        self.eval_env.close()