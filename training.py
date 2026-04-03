import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from g_share import GShare
from typing import Iterable

def to_bit_vector(val: int, num_bits: int) -> torch.Tensor:
    """Converts an integer to a bit-vector tensor using vectorized bitwise operations."""
    val_tensor = torch.tensor(val, dtype=torch.int64)
    shifts = torch.arange(num_bits, dtype=torch.int64)
    return torch.bitwise_and(torch.bitwise_right_shift(val_tensor, shifts), 1).to(torch.float32)

class HashModel(nn.Module):
    def __init__(self, pc: int, history: int, table_size: int):
        super(HashModel, self).__init__()
        
        #generate the distribution
        self.fc1 = nn.Linear((pc+history), 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, table_size)

    def forward(self, pc_tensor: torch.Tensor, history_tensor: torch.Tensor) -> torch.Tensor:
        temp = torch.cat((pc_tensor, history_tensor), dim=-1)
        temp = self.relu(self.fc1(temp))
        logits = self.fc2(temp)
        
        #return probability distribution
        return F.softmax(logits, dim=-1)
    
def training_loop(
        model: nn.Module,
        predictor: GShare,
        optimizer: torch.optim.Optimizer,
        dataloader: Iterable,
        pc_bits: int = 64,
        hist_bits: int = 12,
        device: torch.device = torch.device("cpu"),
        print_interval: int = 1000,
        save_path: str = None,
        save_interval: int = 0):
    
    #train once
    model.train()
    total_loss = 0.0
    steps = 0
    
    for batch in dataloader:
        # Assumes batch processing is sequential to maintain GShare chronology
        pc_raw = batch['pc'].item()
        correct_direction = batch['direction'].item()
        
        pc_val = to_bit_vector(pc_raw, pc_bits).unsqueeze(0).to(device)
        
        masked_history = predictor.hist_vector & ((1 << hist_bits) - 1)
        history_val = to_bit_vector(masked_history, hist_bits).unsqueeze(0).to(device)
        
        #selected the index and get the probability
        probabilities = model(pc_val, history_val)
        distribution = Categorical(probabilities)
        selected_index = distribution.sample()
        log_prob = distribution.log_prob(selected_index)
        
        #poll the branch predictor
        predictor_input = selected_index.item()
        predictor_output = predictor.predict_branch(predictor_input)
        
        predictor.update_predictor(predictor_input, correct_direction)
        
        #calculate loss
        target = (float(correct_direction) * 2.0) - 1.0
        mse_loss_val = (predictor_output - target) ** 2       
        grad_loss = log_prob * float(mse_loss_val)
        
        #back propagation
        optimizer.zero_grad()
        grad_loss.backward()
        optimizer.step()
        
        total_loss += mse_loss_val
        steps += 1
        
        if steps % print_interval == 0:
            print(f"Step {steps:07d} | Current Moving Average MSE: {total_loss / steps:.4f}")
        
        if save_interval > 0 and save_path is not None and steps % save_interval == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path} at step {steps:07d}")
        
    average_mse = total_loss / steps if steps > 0 else 0.0
    return average_mse














































# import torch
# import torch.nn as nn

# def training_loop(
#         predictions: torch.Tensor,
#         correct_list: list,
#         optimizer: torch.optim.Optimizer
# ) -> float:
    
#     #I'm assuming you're giving an unformatted (discrete, 1 for taken and 0 for not taken) list
#     #if not, this step is not needed and the correct_list input needs to be a torch.Tensor
#     correct_tensor = torch.tensor(correct_list, dtype=torch.float32)
#     correct_tensor = (correct_tensor * 2.0) - 1.0
    
#     #this, or the binary CE loss may also work
#     loss = nn.MSELoss(predictions, correct_tensor)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     return loss.item()
