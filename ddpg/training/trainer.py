import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, agent, config, device):
        self.agent = agent
        self.cfg = config
        self.device = device
        self.gamma = config['gamma'] #
        self.tau = config['tau']     #

    def update_policy(self, replay_buffer, batch_size):
        # 1. Sample data
        state, action, next_state, reward, done = replay_buffer.sample(batch_size, self.device)
        
        # (B, H, W, C) -> (B, C, H, W)
        state = state.permute(0, 3, 1, 2).to(self.device) / 255.0
        next_state = next_state.permute(0, 3, 1, 2).to(self.device) / 255.0
        action, reward, done = action.to(self.device), reward.to(self.device), done.to(self.device)

        # 2. Update critic
        with torch.no_grad():
            next_action = self.agent.actor_target(next_state)
            target_q = self.agent.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q = self.agent.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.agent.critic_optimizer.step()

        # 3. Update actor
        actor_loss = -self.agent.critic(state, self.agent.actor(state)).mean()
        
        self.agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.agent.actor_optimizer.step()

        # 4. Soft update network
        self._soft_update(self.agent.actor, self.agent.actor_target)
        self._soft_update(self.agent.critic, self.agent.critic_target)
        
        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)