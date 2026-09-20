# 3. The Teacher (train.py)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
from typing import List, Tuple
from _2_model import AntichessNet

model = AntichessNet()
optimizer = optim.SGD(model.parameters(), lr=0.1)

mse_loss_fn = nn.MSELoss()               
cross_entropy_fn = nn.CrossEntropyLoss() 

def train_step(input_boards: Tensor, actual_values: Tensor, mcts_move_probabilities: Tensor) -> float:
    optimizer.zero_grad()
    predicted_values, predicted_policies = model(input_boards)
    
    value_loss = mse_loss_fn(predicted_values, actual_values)
    policy_loss = cross_entropy_fn(predicted_policies, mcts_move_probabilities)
    
    total_loss = value_loss + (0.2 * policy_loss)
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

def train_loop(training_data: List[Tuple[np.ndarray, np.ndarray, float]], epochs: int = 15, batch_size: int = 256) -> None:
    print(f"Starting training on {len(training_data)} positions...")
    
    for epoch in range(epochs):
        np.random.shuffle(training_data) 
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            
            # THE ULTIMATE BUG FIX: Using tuple unpacking to bypass the interface!
            boards = [b for b, p, r in batch]
            policies = [p for b, p, r in batch]
            rewards = [[r] for b, p, r in batch] 
            
            input_boards = torch.tensor(np.array(boards), dtype=torch.float32)
            mcts_probs = torch.tensor(np.array(policies), dtype=torch.float32)
            actual_values = torch.tensor(np.array(rewards), dtype=torch.float32)
            
            loss = train_step(input_boards, actual_values, mcts_probs)
            
        print(f"Epoch {epoch+1}/{epochs} completed. Total Loss: {loss:.4f}")
