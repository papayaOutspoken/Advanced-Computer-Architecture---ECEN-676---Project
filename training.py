import torch
import torch.nn as nn

def training_loop(
        predictions: torch.Tensor,
        correct_list: list,
        optimizer: torch.optim.Optimizer
) -> float:
    
    #I'm assuming you're giving an unformatted (discrete, 1 for taken and 0 for not taken) list
    #if not, this step is not needed and the correct_list input needs to be a torch.Tensor
    correct_tensor = torch.tensor(correct_list, dtype=torch.float32)
    correct_tensor = (correct_tensor * 2.0) - 1.0
    
    #this, or the binary CE loss may also work
    loss = nn.MSELoss(predictions, correct_tensor)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    return loss.item()