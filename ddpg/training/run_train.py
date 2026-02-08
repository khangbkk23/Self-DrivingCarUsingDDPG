import os
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
from collections import deque
from torch.optim.lr_scheduler import StepLR

from ddpg.training.trainer import Trainer
from ddpg.training.evaluator import Evaluator
from ddpg.agent import DDPGAgent
from ddpg.replay_buffer import ReplayBuffer
from ddpg.utils.visualization import plot_learning_curve, plot_losses
from ddpg.utils.preprocess import ImagePreProcessor

def run_train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(cfg['env_name'], continuous=True)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = DDPGAgent(action_dim, max_action, cfg, device)
    trainer = Trainer(agent, cfg, device)
    evaluator = Evaluator(cfg, device)
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=cfg['buffer_size'])
    processor = ImagePreProcessor()

    actor_scheduler = StepLR(agent.actor_optimizer, step_size=20, gamma=0.5)
    critic_scheduler = StepLR(agent.critic_optimizer, step_size=20, gamma=0.5)
    
    # Global tracking for visualization
    train_rewards = []
    eval_rewards = []
    eval_episodes = []
    actor_losses_history = []
    critic_losses_history = []
    
    best_eval_reward = -np.inf
    patience = cfg['training']['patience']
    patience_counter = 0
    min_delta = cfg['training'].get('min_delta', 0.001)
    
    scores_window = deque(maxlen=20)
    total_steps = 0
    save_path = cfg['training']['save_path']
    plot_path = "./results/plots/"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    # Initialize progress bar
    pbar = tqdm(range(1, cfg['training']['episodes'] + 1), desc="Training")

    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        step_losses_actor = []
        step_losses_critic = []
        
        for t in range(cfg['training']['max_steps_per_episode']):
            total_steps += 1
            
            # Action Selection
            if total_steps < 1000:
                action = env.action_space.sample()
            else:
                state_input = processor.process(state)
                action = agent.select_action(state_input)
                noise = np.random.normal(0, 0.1, size=action_dim)
                action = (action + noise).clip(env.action_space.low, env.action_space.high)

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Optimization
            if replay_buffer.size > cfg['training']['batch_size']:
                c_loss, a_loss = trainer.update_policy(replay_buffer, cfg['training']['batch_size'])
                # Store in step-level for real-time display
                step_losses_critic.append(c_loss)
                step_losses_actor.append(a_loss)
                # Store in global history for visualization
                critic_losses_history.append(c_loss)
                actor_losses_history.append(a_loss)
            
            if done: break
            
        # Logging per episode
        train_rewards.append(episode_reward)
        scores_window.append(episode_reward)
        
        # Calculate mean losses for current episode to show in progress bar
        avg_loss_a = np.mean(step_losses_actor) if step_losses_actor else 0
        avg_loss_c = np.mean(step_losses_critic) if step_losses_critic else 0
        
        pbar.set_postfix({
            "Rew": f"{episode_reward:.1f}",
            "Avg": f"{np.mean(scores_window):.1f}",
            "A_Loss": f"{avg_loss_a:.4f}"
        })

        # Periodic Evaluation and Visualization
        if episode % 5 == 0:
            eval_reward = evaluator.evaluate(agent.actor, n_episodes=3)
            eval_rewards.append(eval_reward)
            eval_episodes.append(episode)
            
            # Use tqdm.write to avoid breaking the progress bar
            tqdm.write(f"\n[Eval] Episode {episode} | Train Avg: {np.mean(scores_window):.1f} | Eval Reward: {eval_reward:.1f}")
            
            actor_scheduler.step()
            critic_scheduler.step()
            
            # Plotting
            plot_learning_curve(train_rewards, eval_rewards, eval_episodes, plot_path)
            plot_losses(actor_losses_history, critic_losses_history, plot_path)
            
            # Early Stopping Check
            if eval_reward > (best_eval_reward + min_delta):
                best_eval_reward = eval_reward
                patience_counter = 0
                torch.save(agent.actor.state_dict(), save_path)
                tqdm.write(f"--> Saved best model (Reward: {best_eval_reward:.2f})")
                
                video_name = f"best_ep_{episode}_reward_{int(eval_reward)}"
                evaluator.save_evaluation_video(agent.actor, filename_prefix=video_name)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                tqdm.write(f"Early stopping triggered at episode {episode}")
                break
                
    env.close()
    evaluator.close()