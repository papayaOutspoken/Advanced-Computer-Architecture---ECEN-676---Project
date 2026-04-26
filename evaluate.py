import argparse
import os
import sys
import itertools
import torch
from torch.utils.data import DataLoader

from training import HashModel, to_bit_vector
from champsim_dataset import ChampSimDataset
import g_share
from g_share import GShare 

def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural Hash vs Baseline GShare")
    parser.add_argument("--trace_file", type=str, required=True, help="Path to the ChampSim .xz trace file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth weights")
    parser.add_argument("--max_steps", type=int, default=-1, help="Limit evaluation steps. -1 for full trace.")
    parser.add_argument("--print_interval", type=int, default=50000, help="Status print interval")
    args = parser.parse_args()

    if not os.path.exists(args.trace_file):
        print(f"Error: Trace file '{args.trace_file}' not found.")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"Error: Model weights '{args.model_path}' not found.")
        sys.exit(1)

    device = torch.device("cpu")
    print("Initialization: Running evaluation on CPU.")
    
    PC_BITS = 64
    HISTORY_BITS = g_share.global_hist_length
    TABLE_SIZE = g_share.hist_table_size

    # Initialize Baseline
    baseline_gshare = GShare()
    
    # Initialize Neural Architecture
    neural_gshare = GShare(hashfn=lambda index, hist: int(index))
    model = HashModel(pc=PC_BITS, history=HISTORY_BITS, table_size=TABLE_SIZE).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() # Lock dropout/batchnorm if any, and set to inference mode

    dataset = ChampSimDataset(args.trace_file)
    dataloader = DataLoader(dataset, batch_size=1)
    if args.max_steps != -1:
        dataloader = itertools.islice(dataloader, args.max_steps)

    print(f"Evaluating {args.model_path} against Baseline on {args.trace_file}...")
    print("-" * 60)

    steps = 0
    base_mispredictions = 0
    neural_mispredictions = 0

    # Disable gradient tracking for pure inference speed
    with torch.no_grad():
        for batch in dataloader:
            pc_raw = batch['pc'].item()
            direction = batch['direction'].item()
            
            # --- BASELINE GSHARE EVALUATION ---
            base_pred_val = baseline_gshare.predict_branch(pc_raw)
            base_pred_dir = 1 if base_pred_val > 0 else 0
            if base_pred_dir != direction:
                base_mispredictions += 1
            baseline_gshare.update_predictor(pc_raw, direction)

            # --- NEURAL GSHARE EVALUATION ---
            pc_val = to_bit_vector(pc_raw, PC_BITS).unsqueeze(0).to(device)
            masked_history = neural_gshare.hist_vector & ((1 << HISTORY_BITS) - 1)
            history_val = to_bit_vector(masked_history, HISTORY_BITS).unsqueeze(0).to(device)
            
            logits = model(pc_val, history_val)
            # Deterministic policy: pick the index with the highest logit
            selected_index = torch.argmax(logits, dim=-1).item()
            
            neural_pred_val = neural_gshare.predict_branch(selected_index)
            neural_pred_dir = 1 if neural_pred_val > 0 else 0
            if neural_pred_dir != direction:
                neural_mispredictions += 1
            neural_gshare.update_predictor(selected_index, direction)

            steps += 1
            
            if steps % args.print_interval == 0:
                base_mpki = (base_mispredictions / steps) * 1000
                neural_mpki = (neural_mispredictions / steps) * 1000
                print(f"Step {steps:08d} | Base MPKI: {base_mpki:.2f} | Neural MPKI: {neural_mpki:.2f}")

    print("-" * 60)
    print("EVALUATION COMPLETE")
    print(f"Total Instructions (Branches): {steps}")
    print(f"Baseline Mispredictions: {base_mispredictions} ({(base_mispredictions/steps)*100:.2f}%)")
    print(f"Neural Mispredictions:   {neural_mispredictions} ({(neural_mispredictions/steps)*100:.2f}%)")
    print(f"Final Baseline MPKI: {(base_mispredictions / steps) * 1000:.3f}")
    print(f"Final Neural MPKI:   {(neural_mispredictions / steps) * 1000:.3f}")
    print("-" * 60)

if __name__ == "__main__":
    main()
