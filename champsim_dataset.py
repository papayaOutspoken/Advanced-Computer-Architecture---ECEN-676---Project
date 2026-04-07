import struct
import subprocess
import parse_champsim_trace
from torch.utils.data import IterableDataset

class ChampSimDataset(IterableDataset):
    def __init__(self, trace_file_path: str):
        super().__init__()
        self.trace_file_path = trace_file_path
        
        # ChampSim traces start with metadata, and each instruction payload is 64-bytes.
        self.struct_format = struct.Struct(parse_champsim_trace.INSTR_FORMAT)
        self.chunk_size = self.struct_format.size

    def __iter__(self):
        # Spawns xzcat to directly pipe the uncompressed binary data to stdout
        process = subprocess.Popen(
            ['xzcat', self.trace_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # 100MB buffer to minimize system call overhead
        )
        
        try:
            while True:
                chunk = process.stdout.read(self.chunk_size)
                if not chunk or len(chunk) < self.chunk_size:
                    break
                
                unpacked_data = self.struct_format.unpack(chunk)
                
                yield {
                    'pc': unpacked_data[0],
                    'direction': unpacked_data[2]
                }
        finally:
            process.stdout.close()
            process.terminate()
            process.wait()