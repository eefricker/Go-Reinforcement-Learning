import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Constants for the 3x3 scale
BOARD_SIZE = 3
HISTORY_LENGTH = 8
INPUT_PLANES = (HISTORY_LENGTH * 2) + 1  # 17 planes

def encode_input_tensor(game):
	input_tensor = np.zeros((INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
	current_player = game.current_player
	opponent = -current_player
	
	# Construct sequence: [Current Board, t-1, t-2, ...]
	boards = [game.board.copy()] + list(reversed(game.history[:-1]))
	
	for i in range(HISTORY_LENGTH):
		if i < len(boards):
			state = boards[i]
			input_tensor[i] = (state == current_player).astype(np.float32)
			input_tensor[i + HISTORY_LENGTH] = (state == opponent).astype(np.float32)
	
	# Color plane
	if current_player == 1:
		input_tensor[16] = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
		
	return input_tensor

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=3, filters=16):
        super().__init__()
        self.board_size = board_size
        num_moves = board_size * board_size + 1
        
        # 1. Stem
        self.stem_conv = nn.Conv2d(17, filters, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(filters)
        
        # 2. Residual Tower
        self.res_block = ResidualBlock(filters)
        
        # 3. Policy Head
        self.p_conv = nn.Conv2d(filters, 2, kernel_size=1)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * board_size * board_size, num_moves)
        
        # 4. Value Head
        self.v_conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(1 * board_size * board_size, 16)
        self.v_fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # Neural net body
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.res_block(x)
        
        # Policy Head output (probabilities via LogSoftmax for training stability)
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        policy = F.softmax(self.p_fc(p), dim=1)
        
        # Value Head output (scalar scalar between -1 and 1)
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v))
        
        return policy, value

def run_neural_net(game, model):
	
    model.eval()  # Switch to evaluation mode (fixes BatchNorm behaviors)
    with torch.no_grad():
		
        input_tensor = encode_input_tensor(game)  # shape (17, 3, 3)
        x = torch.from_numpy(input_tensor).unsqueeze(0).float()
        policy, value = model(x)
      
        return policy[0].numpy(), value[0][0].item()
