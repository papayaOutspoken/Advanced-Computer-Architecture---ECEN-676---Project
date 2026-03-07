#!/usr/bin/env python3
"""
Extract branch only trace from cpb traces.
Based on the official parser: https://github.com/ramisheikh/cbp2025/blob/main/lib/trace_reader.h

From that code:
// Trace Format :
// Inst PC                  - 8 bytes
// Inst Type                - 1 byte
// If load/storeInst
//   Effective Address      - 8 bytes
//   Access Size (total)    - 1 byte
//   Involves Base Update   - 1 byte
//   If Store:
//      Involves Reg Offset - 1 byte
// If branch
//   Taken                  - 1 byte
//   If Taken:
//      Target              - 8 bytes
// Num Input Regs           - 1 byte
// Input Reg Names          - 1 byte each
// Num Output Regs          - 1 byte
// Output Reg Names         - 1 byte each
// Output Reg Values
//   If INT                 - 8 bytes each
//   If SIMD                - 16 bytes each
//
// Int registers are encoded 0-30(GPRs), 31(Stack Pointer Register), 64(Flag Register), 65(Zero Register)
// SIMD registers are encoded 32-63
"""
import struct
import gzip
import sys

# Instruction types (based on the C++ code)
class InstClass:
    UNDEF = 0
    ALU = 1
    LOAD = 2
    STORE = 3
    COND_BRANCH = 4
    UNCOND_DIRECT_BRANCH = 5
    UNCOND_INDIRECT_BRANCH = 6

def is_branch(inst_type):
    return inst_type in [InstClass.COND_BRANCH,
                         InstClass.UNCOND_DIRECT_BRANCH,
                         InstClass.UNCOND_INDIRECT_BRANCH]

def is_mem(inst_type):
    return inst_type in [InstClass.LOAD, InstClass.STORE]

def is_store(inst_type):
    return inst_type == InstClass.STORE

def reg_is_int(reg_offset):
    """Check if register is integer (vs SIMD)"""
    return (reg_offset < 32) or (reg_offset == 64) or (reg_offset == 65)

def read_branch_trace(filename):
    """
    Read trace file and extract branch information
    Returns list of tuples: (pc, taken_flag, target_if_taken)
    """
    branches = []

    # Open file (handle both gzip and regular files)
    if filename.endswith('.gz'):
        f = gzip.open(filename, 'rb')
    else:
        f = open(filename, 'rb')

    try:
        instr_count = 0
        while True:
            # Read PC (8 bytes)
            pc_data = f.read(8)
            if not pc_data or len(pc_data) < 8:
                break  # End of file

            pc = struct.unpack('<Q', pc_data)[0]  # Little-endian 64-bit

            # Read instruction type (1 byte)
            type_data = f.read(1)
            if not type_data:
                break
            inst_type = struct.unpack('B', type_data)[0]

            # Handle memory instructions
            if is_mem(inst_type):
                f.read(8)  # Skip effective address
                f.read(1)  # Skip access size
                f.read(1)  # Skip base update flag
                if is_store(inst_type):
                    f.read(1)  # Skip register offset flag

            # Handle branch instructions
            branch_taken = False
            branch_target = pc + 4  # Default next PC

            if is_branch(inst_type):
                # Read taken flag
                taken_data = f.read(1)
                branch_taken = struct.unpack('B', taken_data)[0] != 0

                if branch_taken:
                    # Read target address
                    target_data = f.read(8)
                    branch_target = struct.unpack('<Q', target_data)[0]

                # Add to branches list
                branches.append((pc, branch_taken, branch_target))

            # Read register information
            # Number of input registers
            num_in_regs = struct.unpack('B', f.read(1))[0]
            # Input register names
            for _ in range(num_in_regs):
                f.read(1)

            # Number of output registers
            num_out_regs = struct.unpack('B', f.read(1))[0]
            # Output register names
            out_regs = []
            for _ in range(num_out_regs):
                out_reg = struct.unpack('B', f.read(1))[0]
                out_regs.append(out_reg)

            # Output register values
            for out_reg in out_regs:
                if reg_is_int(out_reg):
                    f.read(8)  # 8 bytes for INT
                else:
                    f.read(16)  # 16 bytes for SIMD

            instr_count += 1
            if instr_count % 1000000 == 0:
                print(f"Processed {instr_count} instructions...")

    finally:
        f.close()

    return branches

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <trace_file>")
        sys.exit(1)

    trace_file = sys.argv[1]
    print(f"Reading trace file: {trace_file}")

    branches = read_branch_trace(trace_file)

    print(f"\nFound {len(branches)} branches")
    print("\nFirst 10 branches:")
    print("PC\t\tTaken\tTarget")
    print("-" * 40)

    for i, (pc, taken, target) in enumerate(branches[:10]):
        taken_str = "T" if taken else "N"
        print(f"0x{pc:016x}\t{taken_str}\t0x{target:016x}")

    # Calculate statistics
    if branches:
        taken_count = sum(1 for _, taken, _ in branches if taken)
        print(f"\nBranch Statistics:")
        print(f"Total branches: {len(branches)}")
        print(f"Taken: {taken_count} ({100*taken_count/len(branches):.2f}%)")
        print(f"Not taken: {len(branches)-taken_count} ({100*(len(branches)-taken_count)/len(branches):.2f}%)")

    return branches

if __name__ == "__main__":
    main()