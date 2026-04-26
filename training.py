import torch
import torch.nn as nn
from torch.distributions import Categorical
from g_share import GShare
from typing import Iterable

def to_bit_vector(val: int, num_bits: int) -> torch.Tensor:
    """Converts an integer to a bit-vector tensor using vectorized bitwise operations."""
    val_tensor = torch.tensor(val, dtype=torch.int64)
    shifts = torch.arange(num_bits, dtype=torch.int64)
    return torch.bitwise_and(torch.bitwise_right_shift(val_tensor, shifts), 1).to(torch.float32)

def _ints_to_bit_matrix(vals: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Convert int tensor [B] -> float32 bit matrix [B, num_bits] on the same device."""
    # Bits are least-significant-first
    shifts = torch.arange(num_bits, device=vals.device, dtype=torch.int32)
    return ((vals.to(torch.int64).unsqueeze(1) >> shifts) & 1).to(torch.float32)

def _discounted_return_to_go(costs: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted return-to-go for a cost sequence.

    Given costs c[t], returns G[t] = sum_{k=t..T-1} gamma^(k-t) * c[k].
    Implemented with vectorized torch ops (no Python loop).
    """
    if costs.numel() == 0:
        return costs
    if gamma == 1.0:
        return torch.flip(torch.cumsum(torch.flip(costs, dims=(0,)), dim=0), dims=(0,))

    t = torch.arange(costs.shape[0], device=costs.device, dtype=costs.dtype)
    discounts = torch.pow(torch.tensor(gamma, device=costs.device, dtype=costs.dtype), t)
    discounted_costs = costs * discounts
    rev_cumsum = torch.flip(torch.cumsum(torch.flip(discounted_costs, dims=(0,)), dim=0), dims=(0,))
    return rev_cumsum / discounts

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
        # Return unnormalized logits for numerical stability. Callers can use
        # Categorical(logits=...) or softmax as needed.
        return logits

def training_loop(
        model: nn.Module,
        predictor: GShare,
        optimizer: torch.optim.Optimizer,
        dataloader: Iterable,
        pc_bits: int = 64,
        hist_bits: int = 14,
        device: torch.device = torch.device("cpu"),
        print_interval: int = 1000,
        save_path: str = None,
        save_interval: int = 0,
        rollout_steps: int = 2**8,
        gamma: float = 0.997, # reinforcement learning future discounting factor
        baseline_beta: float = 0.05,
        metric_window: int = 10000,
    ):

    #train once
    model.train()
    total_loss = 0.0
    steps = 0
    correct_predictions = 0

    metrics_file = "standard_training_metrics.csv"
    with open(metrics_file, mode='w', newline='') as f:
        f.write("Step,Average_MSE,Accuracy,Window_MSE,Window_Accuracy\n")

    # Precompute history bit-vectors: only 2^hist_bits possible.
    # This removes per-step bitwise conversion work for history.
    history_lut = _ints_to_bit_matrix(
        torch.arange(1 << hist_bits, device=device, dtype=torch.int64),
        hist_bits,
    )

    rollout_log_probs = []
    rollout_mse = []
    baseline_ema = None  # scalar float baseline for advantage estimation

    # Rolling window metrics help distinguish true training regressions from
    # non-stationarity in the trace.
    win_mse_sum = 0.0
    win_correct = 0
    win_steps = 0

    dataloader_iter = iter(dataloader)
    while True:
        # Collect a rollout window worth of (pc, history, direction) first.
        # This lets us run the policy network as a single batch on GPU.
        pc_raws = []
        hist_raws = []
        directions = []

        for _ in range(rollout_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break

            # Support both dict samples and DataLoader-collated dict batches.
            if isinstance(batch, dict):
                pc_raw = int(batch['pc']) if not hasattr(batch['pc'], 'item') else batch['pc'].item()
                correct_direction = int(batch['direction']) if not hasattr(batch['direction'], 'item') else batch['direction'].item()
            else:
                # Unexpected sample format.
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            pc_raws.append(pc_raw)
            hist_raws.append(predictor.hist_vector & ((1 << hist_bits) - 1))
            directions.append(int(correct_direction))

        if not pc_raws:
            break

        pc_raws_t = torch.tensor(pc_raws, dtype=torch.int64, device=device)
        pc_bits_t = _ints_to_bit_matrix(pc_raws_t, pc_bits)
        hist_idx_t = torch.tensor(hist_raws, dtype=torch.int64, device=device)
        history_bits_t = history_lut[hist_idx_t]

        logits_t = model(pc_bits_t, history_bits_t)  # [T, table_size]
        dist = Categorical(logits=logits_t)
        selected_idx_t = dist.sample()  # [T]
        log_prob_t = dist.log_prob(selected_idx_t)  # [T]

        # Step through the environment sequentially on CPU while consuming sampled actions.
        for i in range(len(pc_raws)):
            predictor_input = int(selected_idx_t[i].item())
            correct_direction = directions[i]

            predictor_output = predictor.predict_branch(predictor_input)

            pred_dir = 1 if predictor_output > 0 else 0
            if pred_dir == correct_direction:
                correct_predictions += 1
                win_correct += 1

            predictor.update_predictor(predictor_input, correct_direction)

            target = (float(correct_direction) * 2.0) - 1.0
            mse_loss_val = (predictor_output - target) ** 2
            win_mse_sum += float(mse_loss_val)
            win_steps += 1

            rollout_log_probs.append(log_prob_t[i])
            rollout_mse.append(float(mse_loss_val))

            total_loss += mse_loss_val
            steps += 1

        # Policy gradient update at end of rollout (or partial rollout).
        if rollout_mse:
            costs_t = torch.tensor(rollout_mse, dtype=torch.float32, device=device)
            returns_t = _discounted_return_to_go(costs_t, gamma)
            log_probs_t = torch.stack(rollout_log_probs)

            rollout_mean_return = float(returns_t.mean().item())
            if baseline_ema is None:
                baseline_ema = rollout_mean_return
            else:
                baseline_ema = (1.0 - baseline_beta) * baseline_ema + baseline_beta * rollout_mean_return

            advantage_t = returns_t - float(baseline_ema)
            pg_loss = torch.sum(log_probs_t * advantage_t)

            optimizer.zero_grad()
            pg_loss.backward()
            optimizer.step()

            rollout_log_probs.clear()
            rollout_mse.clear()

        if steps % print_interval == 0:
            current_avg_mse = total_loss / steps
            current_accuracy = correct_predictions / steps
            w_mse = (win_mse_sum / win_steps) if win_steps else 0.0
            w_acc = (win_correct / win_steps) if win_steps else 0.0
            ema_str = f"{baseline_ema:.4f}" if baseline_ema is not None else "n/a"
            print(
                f"Step {steps:07d} | Avg MSE: {current_avg_mse:.4f} | Acc: {current_accuracy:.4f} "
                f"| Window({metric_window}) MSE: {w_mse:.4f} | Acc: {w_acc:.4f} "
                f"| Baseline(EMA): {ema_str}"
            )
            with open(metrics_file, mode='a', newline='') as f:
                f.write(
                    f"{steps},{current_avg_mse:.4f},{current_accuracy:.4f},"
                    f"{w_mse:.4f},{w_acc:.4f}\n"
                )

            # Reset window on each print so it reflects recent behavior.
            win_mse_sum = 0.0
            win_correct = 0
            win_steps = 0

        if save_interval > 0 and save_path is not None and steps % save_interval == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path} at step {steps:07d}")

    average_mse = total_loss / steps if steps > 0 else 0.0
    return average_mse
