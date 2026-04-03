import argparse
import os
import sys
import itertools
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from training import HashModel, training_loop, to_bit_vector
from champsim_dataset import ChampSimDataset
from g_share import GShare 

def main():
    parser = argparse.ArgumentParser(description="Train Neural Hash Model on ChampSim Traces")
    parser.add_argument("--trace_file", type=str, required=True, help="Path to the ChampSim .xz trace file")
    parser.add_argument("--max_steps", type=int, default=-1, help="Number of trace instructions to process. Set to -1 for the entire trace.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the Adam optimizer")
    parser.add_argument("--print_interval", type=int, default=5000, help="How often to print MSE to standard output")
    parser.add_argument("--save_path", type=str, default="neural_hash_model.pth", help="Path to save the trained model weights")
    parser.add_argument("--save_interval", type=int, default=500000, help="How often to save a model checkpoint (in steps). Set to 0 to disable periodic saving.")
    
    args = parser.parse_args()

    if not os.path.exists(args.trace_file):
        print(f"Error: Trace file '{args.trace_file}' not found.")
        sys.exit(1)

    # Hardware Configuration
    # We explicitly force the CPU. Because GShare requires strictly sequential processing 
    # (batch_size=1), the PCI-e transfer overhead of moving single instructions to a GPU 
    # is significantly slower than executing the operations directly on the CPU.
    device = torch.device("cpu")
    print(f"Initialization: Forcing compute device -> {device} (Optimized for sequential batch_size=1)")
    print(f"Target Trace: {args.trace_file}")
    print(f"Maximum Steps: {'Unlimited (Entire Trace)' if args.max_steps == -1 else args.max_steps}")

    # Architecture Constraints
    PC_BITS = 64
    HISTORY_BITS = 12
    TABLE_SIZE = 16384 

    # Initialize Architecture Components
    gshare_predictor = GShare()
    
    # Initialize the PyTorch Neural Hash Model. 
    model = HashModel(pc=PC_BITS, history=HISTORY_BITS, table_size=TABLE_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Initialized GShare, Neural Hash Model, and Adam Optimizer successfully.")

    # Initialize Data Pipeline
    print("Spawning xzcat subprocess for sequential data streaming...")
    dataset = ChampSimDataset(args.trace_file)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Slice the dataloader to prevent reading the entire multi-billion instruction file during tests
    if args.max_steps != -1:
        dataloader_to_use = itertools.islice(dataloader, args.max_steps)
    else:
        dataloader_to_use = dataloader

    print("-" * 50)
    print("Starting Training Loop...")
    print("-" * 50)

    # Execute Training
    average_mse = training_loop(
        model=model,
        predictor=gshare_predictor,
        optimizer=optimizer,
        dataloader=dataloader_to_use,
        pc_bits=PC_BITS,
        hist_bits=HISTORY_BITS,
        device=device,
        print_interval=args.print_interval,
        save_path=args.save_path,
        save_interval=args.save_interval
    )

    print("-" * 50)
    print(f"Training complete. Final Average MSE: {average_mse:.4f}")

    # Post-Execution State Verification
    has_gradients = any(param.grad is not None for param in model.parameters())
    if has_gradients:
        print("Verification: Model parameters contain gradients. Backpropagation was successful.")
    else:
        print("Verification Error: No gradients found in model parameters. The backward pass failed.")

    # Save the trained model checkpoint
    print(f"\nSaving model state dictionary to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)
    print("Done.")

if __name__ == "__main__":
    main()