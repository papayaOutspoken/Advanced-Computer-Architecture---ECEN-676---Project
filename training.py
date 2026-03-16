import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from g_share import GShare

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
        pc_stream: list,
        canonical_list: list):
    
    #train once
    model.train()
    total_loss = 0.0
    
    for i in range(len(pc_stream)):
        pc_val = torch.tensor([pc_stream[i]], dtype=torch.float32)
        history_val = torch.tensor([predictor.hist_vector], dtype=torch.float32)
        
        #selected the index and get the probability
        probabilities = model(pc_val, history_val)
        distribution = Categorical(probabilities)
        selected_index = distribution.sample()
        log_prob = distribution.log_prob(selected_index)
        
        #poll the branch predictor
        predictor_input = selected_index.item()
        predictor_output = predictor.predict_branch(predictor_input)
        
        correct_direction = canonical_list[i]
        predictor.update_predictor(predictor_input, correct_direction)
        
        #calculate loss
        target = (float(correct_direction) * 2.0) - 1.0
        mse_loss_val = (predictor_output - target) ** 2       
        grad_loss = torch.tensor(mse_loss_val, requires_grad=False) * -log_prob
        
        #back propagation
        optimizer.zero_grad()
        grad_loss.backward()
        optimizer.step()
        
        total_loss += mse_loss_val
        
    average_mse = total_loss/len(pc_stream)
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
