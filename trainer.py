import numpy as np
import datetime as dt
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment import RealGoGame
from mcts import run_mcts
from model import encode_input_tensor

class AlphaZeroTrainer:
    def __init__(self, model, learning_rate=0.01, weight_decay=1e-4, sgd_momentum=0.9):
        self.model = model
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=learning_rate, 
            momentum=sgd_momentum, 
            weight_decay=weight_decay
        )
        
    def train_step(self, x_batch, pi_targets, v_targets):
        self.model.train()  # Switch network to training mode (important for BatchNorm updates)
        self.optimizer.zero_grad()
        
        # 1. Convert NumPy arrays from the ReplayBuffer into PyTorch Tensors
        inputs = torch.from_numpy(x_batch).float()
        target_pis = torch.from_numpy(pi_targets).float()
        target_vs = torch.from_numpy(v_targets).float()
        
        # 2. Forward Pass
        pred_pis, pred_vs = self.model(inputs)
        
        # 3. Calculate AlphaZero Losses
        loss_v = F.mse_loss(pred_vs, target_vs)
        # Policy Loss: Cross Entropy (-sum(target * log(pred)))
        loss_p = -torch.mean(torch.sum(target_pis * torch.log(pred_pis + 1e-7), dim=1))
        
        total_loss = loss_v + loss_p
        
        # 4. Backward Pass & Parameter Step
        total_loss.backward()
        
        # Automatic gradient clipping
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss_v.item(), loss_p.item()


class ReplayBuffer:
    def __init__(self, max_size=2000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state, pi, v):
        self.buffer.append((state, pi, v))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)


def complete_training_loop(model, num_games=500, batch_size=32, training_steps_per_game=4):
    """
    Executes full AlphaZero pipeline: Self-Play Generation -> Replay Buffering -> PyTorch Optimization.
    Optimizes the 'model' parameters directly in-place.
    """
    trainer = AlphaZeroTrainer(model, learning_rate=0.01, weight_decay=1e-4)
    replay_buffer = ReplayBuffer(max_size=2000)
    
    tic = dt.datetime.today()
    print(f"Starting AlphaZero training loop for {num_games} games...")
    
    for i in range(num_games):
        game = RealGoGame(board_size=3)
        game_history = []  # Stores: (encoded_state, mcts_pi, player_at_turn)
        
        # --- PHASE 1: EXECUTE SELF-PLAY ---
        # The game actually plays out here, using MCTS guided by the PyTorch model
        while not game.is_game_over():
            # Get target policy distribution from MCTS
            pi, _ = run_mcts(game, model, num_simulations=50)
            
            # Record the state from the current player's perspective
            encoded_state = encode_input_tensor(game)
            game_history.append((encoded_state, pi, game.current_player))
            
            # Select move based on MCTS search probabilities
            moves = game.get_valid_moves()
            
            move_weights = []
            for move in moves:
                if move == "PASS":
                    idx = 9  # game.board_size**2
                else:
                    idx = move[0] * 3 + move[1]
                move_weights.append(pi[idx])
                
            action = random.choices(moves, weights=move_weights)[0]
            
            # Execute the move on the board
            game.step(action)
            
        # --- PHASE 2: SCORE REWARDS & BUFFER DATA ---
        game_winner = game.get_reward()  # 1 for Black win, -1 for White win, 0 for Draw
        for state, pi, player_at_turn in game_history:
            # Value target is 1 if the player who made the move won, -1 if they lost
            value_target = game_winner * player_at_turn
            replay_buffer.add(state, pi, value_target)
            
        # Ensure enough games to draw a representative batch
        if len(replay_buffer) < batch_size:
            if i % 10 == 0 or i == num_games - 1:
                print(f"Game {i+1}: Warming up replay buffer... ({len(replay_buffer)}/{batch_size} states)")
            continue
            
        # --- PHASE 3: GRADIENT DESCENT OPTIMIZATION ---
        total_loss_v, total_loss_p = 0.0, 0.0
        for _ in range(training_steps_per_game):
            batch = replay_buffer.sample(batch_size)
            states, policies, values = zip(*batch)
            
            x_batch = np.array(states)               
            pi_batch = np.array(policies)            
            v_batch = np.array(values).reshape(-1, 1)
            
            # Run a step of PyTorch backpropagation
            loss_v, loss_p = trainer.train_step(x_batch, pi_batch, v_batch)
            
            total_loss_v += loss_v
            total_loss_p += loss_p
            
        # Diagnostic prints
        if i % 50 == 49:
            avg_loss_v = total_loss_v / training_steps_per_game
            avg_loss_p = total_loss_p / training_steps_per_game
            time_elapsed = dt.datetime.today() - tic
            s_per_game = time_elapsed.seconds / (i + 1)
            print(f"Game {i+1}/{num_games} | Value Loss: {avg_loss_v:.4f} | Policy Loss: {avg_loss_p:.4f} | Pace: {s_per_game:.2f}s/game")
            
    print("Training Complete!")
