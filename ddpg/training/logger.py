import os
import csv
import time
from datetime import datetime

class Logger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.log_file_path = os.path.join(self.save_dir, 'training_log.csv')
        self.header_written = False
        self.file_handle = open(self.log_file_path, 'w', newline='')
        self.writer = None
        
        print(f"[Logger] Logging to: {self.log_file_path}")

    def log(self, metrics):
        if not self.header_written:
            self.writer = csv.DictWriter(self.file_handle, fieldnames=metrics.keys())
            self.writer.writeheader()
            self.header_written = True
            
        self.writer.writerow(metrics)
        self.file_handle.flush()
        
    def print_terminal(self, metrics):
        now = datetime.now().strftime("%H:%M:%S")
        
        ep = metrics.get('episode', 0)
        rew = metrics.get('reward', 0)
        avg_rew = metrics.get('avg_reward', 0)
        a_loss = metrics.get('actor_loss', 0)
        c_loss = metrics.get('critic_loss', 0)
        step = metrics.get('steps', 0)
        
        print(f"[{now}] Episode {ep:04d} | Step {step:06d} | "
              f"Rew: {rew:6.1f} | Avg: {avg_rew:6.1f} | "
              f"A_Loss: {a_loss:7.4f} | C_Loss: {c_loss:7.4f}")

    def close(self):
        if self.file_handle:
            self.file_handle.close()