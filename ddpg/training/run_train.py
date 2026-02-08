import os
import gym
import torch
import yaml
import numpy as np
from collections import deque
from torch.optim.lr_scheduler import StepLR

from ddpg.models.agent import DDPGAgent
from ddpg.models.replay_buffer import ReplayBuffer
from training.trainer import Trainer
from training.evaluator import Evaluator

def load_config(path='./ddpg/config/env.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_train():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make(cfg['env'], continuous=True)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = DDPGAgent(action_dim, max_action, cfg, device)
    trainer = Trainer(agent, cfg, device)
    evaluator = Evaluator(cfg, device)
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=cfg['buffer_size'])
    
    # scheduler
    actor_scheduler = StepLR(agent.actor_optimizer, step_size=20, gamma=0.5)
    critic_scheduler = StepLR(agent.critic_optimizer, step_size=20, gamma=0.5)

	# early stopping
    best_eval_reward = -np.inf
    patience = cfg['training']['patience']
    patience_counter = 0
    min_delta = cfg['training'].get('min_delta', 0.001)
    
    scores_window = deque(maxlen=20)
    total_steps = 0
    save_path = cfg['training']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for episode in range(1, cfg['training']['epoch'] + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(cfg['training']['max_steps_per_episode']):
            total_steps += 1
            
            if total_steps < 1000:
                action = env.action_space.sample()
            else:
                state_input = state.transpose(2, 0, 1)
                action = agent.select_action(state_input)

                noise = np.random.normal(0, 0.1, size=action_dim)
                action = (action + noise).clip(env.action_space.low, env.action_space.high)

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            # store experience in buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if replay_buffer.size > cfg['training']['batch_size']:
                trainer.update_policy(replay_buffer, cfg['training']['batch_size'])
            
            if done: break
            
        scores_window.append(episode_reward)
        avg_train_score = np.mean(scores_window)

        if episode % 5 == 0:
            eval_reward = evaluator.evaluate(agent.actor, n_episodes=3)
            print(f"Episode {episode} | Train: {avg_train_score:.1f} | Eval: {eval_reward:.1f}")
            

            actor_scheduler.step()
            critic_scheduler.step()
            
            if eval_reward > (best_eval_reward + min_delta):
                best_eval_reward = eval_reward
                patience_counter = 0
                torch.save(agent.actor.state_dict(), save_path)
                print(f"--> Saved best model with reward: {best_eval_reward:.2f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at episode {episode}")
                break
                
    env.close()
    evaluator.close()

if __name__ == "__main__":
    run_train()