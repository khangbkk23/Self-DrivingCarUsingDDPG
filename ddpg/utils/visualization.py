import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curve(train_rewards, eval_rewards, eval_episodes, save_path):
    """Plot training and evaluation rewards over episodes."""
    plt.figure(figsize=(10, 6))
    
    # Plot Training Reward
    plt.plot(train_rewards, label='Training Reward', alpha=0.3, color='blue')
    if len(train_rewards) >= 10:
        moving_avg = np.convolve(train_rewards, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(train_rewards)), moving_avg, label='Train Reward (MA 10)', color='navy')
    
    # Plot Evaluation Reward
    plt.scatter(eval_episodes, eval_rewards, color='red', label='Eval Reward')
    plt.plot(eval_episodes, eval_rewards, color='red', linestyle='dashed', alpha=0.5)
    
    plt.title('DDPG Learning Curve - CarRacing-v2')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(os.path.join(save_path, 'learning_curve.png'))
    plt.close()

def plot_losses(actor_losses, critic_losses, save_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Critic Loss (Primary Y-axis)
    ax1.set_xlabel('Optimization Steps')
    ax1.set_ylabel('Critic Loss (MSE)', color='tab:red')
    ax1.plot(critic_losses, color='tab:red', alpha=0.6, label='Critic Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Actor Loss (Secondary Y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Actor Loss (Negative Q)', color='tab:blue')
    ax2.plot(actor_losses, color='tab:blue', alpha=0.6, label='Actor Loss')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('DDPG Optimization Losses')
    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(os.path.join(save_path, 'training_losses.png'))
    plt.close()