import argparse
import os
import sys
import itertools
import torch
import torch.nn as nn
from evotorch import Problem
from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger

from training import HashModel, to_bit_vector
from champsim_dataset import ChampSimDataset
from g_share import GShare

def main():
    parser = argparse.ArgumentParser(description="EvoTorch Training for Neural Hash Model")
    parser.add_argument("--trace_file", type=str, required=True, help="Path to ChampSim trace file")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of instructions to load into memory for evaluation")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size for EvoTorch")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to run")
    parser.add_argument("--save_path", type=str, default="neural_hash_evotorch.pth", help="Path to save the best model")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Ray workers to spawn (decrease if OOM)")
    args = parser.parse_args()

    if not os.path.exists(args.trace_file):
        print(f"Error: Trace file '{args.trace_file}' not found.")
        sys.exit(1)

    print(f"Loading {args.chunk_size} instructions from {args.trace_file} into memory...")
    dataset = ChampSimDataset(args.trace_file)
    
    # Simplify the chunk to plain tuples to minimize pickling/serialization overhead for Ray workers
    trace_chunk = []
    for step in itertools.islice(dataset, args.chunk_size):
        trace_chunk.append((step['pc'], step['direction']))
    
    print("Done loading chunk. Setting up EvoTorch Problem...")

    PC_BITS = 64
    HISTORY_BITS = 12
    TABLE_SIZE = 16384

    # Create a dummy model to calculate the total number of weights/parameters
    dummy_model = HashModel(pc=PC_BITS, history=HISTORY_BITS, table_size=TABLE_SIZE)
    num_params = sum(p.numel() for p in dummy_model.parameters())
    print(f"Neural Hash Model has {num_params} parameters.")

    print("Precomputing PC and History bit-vectors to save memory and compute...")
    # Precompute PC tensors to eliminate in-loop allocations
    pc_tensors = torch.stack([to_bit_vector(pc, PC_BITS) for pc, _ in trace_chunk])
    directions = [direction for _, direction in trace_chunk]
    # Precompute all 4096 possible history combinations
    history_tensors = torch.stack([to_bit_vector(h, HISTORY_BITS) for h in range(1 << HISTORY_BITS)])

    # The Fitness Function evaluated by EvoTorch Ray Actors
    def evaluate_solution(weights: torch.Tensor) -> torch.Tensor:
        # 1. Reconstruct the PyTorch model from the 1D array of weights
        model = HashModel(pc=PC_BITS, history=HISTORY_BITS, table_size=TABLE_SIZE)
        nn.utils.vector_to_parameters(weights, model.parameters())
        model.eval() # Deterministic inference mode
        
        # 2. Setup GShare. We must use a passthrough hashfn because our neural network 
        # is providing the index directly, bypassing the traditional XOR.
        gshare = GShare(hashfn=lambda index, hist: int(index))
        correct_predictions = 0
        
        # 3. Simulate the branch predictor sequence without tracking gradients
        with torch.no_grad():
            for step_idx in range(len(directions)):
                direction = directions[step_idx]
                pc_val = pc_tensors[step_idx].unsqueeze(0)
                masked_history = gshare.hist_vector & ((1 << HISTORY_BITS) - 1)
                history_val = history_tensors[masked_history].unsqueeze(0)
                
                # Neural network outputs probabilities over all 16384 indices
                probabilities = model(pc_val, history_val)
                selected_index = torch.argmax(probabilities, dim=-1).item()
                
                # Poll the GShare internal table
                pred_val = gshare.raw_predict_branch(selected_index)
                pred_dir = 1 if pred_val > 0 else 0
                
                if pred_dir == direction:
                    correct_predictions += 1
                    
                gshare.update_predictor(selected_index, direction)
                
        # Fitness is simply the number of correct predictions (EvoTorch will maximize this)
        return torch.tensor(correct_predictions, dtype=torch.float32)

    # Define the problem for EvoTorch
    problem = Problem("max", evaluate_solution, solution_length=num_params, initial_bounds=(-1.0, 1.0), num_actors=args.num_workers)

    print("Initializing PGPE Searcher...")
    # Extract the initialized weights from our PyTorch model to serve as the search center
    center_init = nn.utils.parameters_to_vector(dummy_model.parameters()).detach()
    searcher = PGPE(problem, popsize=args.pop_size, center_learning_rate=0.05, stdev_learning_rate=0.1, stdev_init=0.1, optimizer="clipup", center_init=center_init)

    # This logger will print Min, Max, and Mean fitness per generation
    logger = StdOutLogger(searcher)

    print("-" * 50)
    print(f"Starting Evolutionary Training for {args.generations} generations...")
    searcher.run(args.generations)

    print("-" * 50)
    print(f"Saving the center point of the search distribution to {args.save_path}...")
    center_solution = searcher.status["center"]
    nn.utils.vector_to_parameters(center_solution, dummy_model.parameters())
    torch.save(dummy_model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()