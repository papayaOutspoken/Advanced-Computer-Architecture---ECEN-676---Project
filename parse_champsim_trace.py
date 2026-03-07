#!/usr/bin/env python3
"""
Parses out branch traces from champsim format traces.

A champsim trace (is currently) defined as many c structs instances dumped with memcpy to disk of this form:

```c
// special registers that help us identify branches
namespace champsim
{
constexpr char REG_STACK_POINTER = 6;
constexpr char REG_FLAGS = 25;
constexpr char REG_INSTRUCTION_POINTER = 26;
} // namespace champsim

// instruction format
constexpr std::size_t NUM_INSTR_DESTINATIONS_SPARC = 4;
constexpr std::size_t NUM_INSTR_DESTINATIONS = 2;
constexpr std::size_t NUM_INSTR_SOURCES = 4;

struct input_instr {
  // instruction pointer or PC (Program Counter)
  unsigned long long ip;

  // branch info
  unsigned char is_branch;j
  unsigned char branch_taken;

  unsigned char destination_registers[NUM_INSTR_DESTINATIONS]; // output registers
  unsigned char source_registers[NUM_INSTR_SOURCES];           // input registers

  unsigned long long destination_memory[NUM_INSTR_DESTINATIONS]; // output memory
  unsigned long long source_memory[NUM_INSTR_SOURCES];           // input memory
};
```
"""

import struct
import sys

# Constants matching the C++ structure
NUM_INSTR_DESTINATIONS = 2
NUM_INSTR_SOURCES = 4

# Structure format string for unpacking
# Q = unsigned long long (8 bytes)
# B = unsigned char (1 byte)
# Format: ip(Q) + is_branch(B) + branch_taken(B) +
#         dest_regs(2B) + src_regs(4B) +
#         dest_mem(2Q) + src_mem(4Q)
INSTR_FORMAT = '<QBB2B4B2Q4Q'
INSTR_SIZE = struct.calcsize(INSTR_FORMAT)  # Should be 64 bytes

def parse_trace_file(filename: str) -> list[tuple[int, bool]]:
    """
    Parse a ChampSim trace file and extract branch information.

    Args:
        filename: Path to the binary trace file

    Returns:
        List of tuples containing (branch_address, taken_flag)
        where branch_address is the instruction pointer and
        taken_flag is True if branch was taken, False otherwise
    """
    branches = []

    try:
        with open(filename, 'rb') as f:
            while True:
                # Read one instruction structure
                data = f.read(INSTR_SIZE)
                if not data or len(data) < INSTR_SIZE:
                    break

                # Unpack the binary data
                unpacked = struct.unpack(INSTR_FORMAT, data)

                # Extract relevant fields
                ip = unpacked[0]
                is_branch = unpacked[1]
                branch_taken = unpacked[2]

                # If this is a branch instruction, add it to our list
                if is_branch == 1:
                    branches.append((ip, bool(branch_taken)))

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    return branches

def main():
    """Main function to demonstrate usage."""
    if len(sys.argv) != 2:
        print("Usage: python parse_trace.py <trace_file>")
        sys.exit(1)

    filename = sys.argv[1]
    branches = parse_trace_file(filename)

    # Print summary
    print(f"Total branches found: {len(branches)}")
    if branches:
        taken_count = sum(1 for _, taken in branches if taken)
        not_taken_count = len(branches) - taken_count
        print(f"Taken: {taken_count} ({taken_count/len(branches)*100:.2f}%)")
        print(f"Not taken: {not_taken_count} ({not_taken_count/len(branches)*100:.2f}%)")

        # Print first 10 branches as examples
        print("\nFirst 10 branches:")
        for i, (addr, taken) in enumerate(branches[:10]):
            status = "TAKEN" if taken else "NOT_TAKEN"
            print(f"  {i+1:3d}. 0x{addr:016x} -> {status}")

    return branches

if __name__ == "__main__":
    branches = main()
