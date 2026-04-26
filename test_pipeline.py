import argparse
import os
import sys
import itertools
import torch
import torch.optim as optim

from training import HashModel, training_loop
from champsim_dataset import ChampSimDataset
import g_share
from g_share import GShare

def main():
    parser = argparse.ArgumentParser(description="Train Neural Hash Model on ChampSim Traces")
    parser.add_argument("--trace_file", type=str, required=True, help="Path to the ChampSim .xz trace file")
    parser.add_argument("--max_steps", type=int, default=-1, help="Number of trace instructions to process. Set to -1 for the entire trace.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the Adam optimizer")
    parser.add_argument("--print_interval", type=int, default=5000, help="How often to print MSE to standard output")
    parser.add_argument("--save_path", type=str, default="neural_hash_model.pth", help="Path to save the trained model weights")
    parser.add_argument("--save_interval", type=int, default=500000, help="How often to save a model checkpoint (in steps). Set to 0 to disable periodic saving.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device for the neural policy network")

    args = parser.parse_args()

    if not os.path.exists(args.trace_file):
        print(f"Error: Trace file '{args.trace_file}' not found.")
        sys.exit(1)

    # Hardware Configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Initialization: Policy network device -> {device}")
    print(f"Target Trace: {args.trace_file}")
    print(f"Maximum Steps: {'Unlimited (Entire Trace)' if args.max_steps == -1 else args.max_steps}")

    # Architecture Constraints
    PC_BITS = 64

    # Initialize Architecture Components
    gshare_predictor = GShare(hashfn=lambda index, hist: int(index))

    # Initialize the PyTorch Neural Hash Model.
    model = HashModel(pc=PC_BITS, history=g_share.global_hist_length, table_size=g_share.hist_table_size).to(device)
    # See <https://www.reddit.com/r/MachineLearning/comments/qq75zu/d_how_do_you_choose_an_optimizer_and_why_are/>
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, amsgrad=True, maximize=False)

    print("Initialized GShare, Neural Hash Model, and Adam Optimizer successfully.")

    # Initialize Data Pipeline
    print("Spawning xzcat subprocess for sequential data streaming...")
    dataset = ChampSimDataset(args.trace_file)

    # Slice the stream to prevent reading the entire multi-billion instruction file during tests
    if args.max_steps != -1:
        dataloader_to_use = itertools.islice(dataset, args.max_steps)
    else:
        dataloader_to_use = dataset

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
        hist_bits=g_share.global_hist_length,
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
