global_hist_length = 14
counter_bits = 2
hist_table_size = 16384


class GShare:
    def __init__(self, hashfn: callable[[int,int], int] = lambda addr, hist: addr ^ hist):
        self.hist_table = [2**(counter_bits-1)] * hist_table_size # weakly taken start
        self.hist_vector = 0 # Will have to manually make sure this is less than num bits set in global_hist_length
        self.hashfn = hashfn

    def predict_branch(self, address: int): #output will be normalized from 1 to -1
        return self.raw_predict_branch(self.hashfn(address, self.hist_vector))

    def raw_predict_branch(self, idx: int):
        value = self.hist_table[idx % hist_table_size]
        # Map [0, 2^n - 1] to [-1, 1]
        max_val = (2 ** counter_bits) - 1
        return (2 * value - max_val) / max_val

    def update_predictor(self, address, result): # result expected to be 1 for taken, 0 or -1 for not taken
        index = self.hashfn(address, self.hist_vector) % hist_table_size
        
        if result == 1:  # taken
            if self.hist_table[index] < (2**counter_bits - 1):
                self.hist_table[index] += 1
        else:  # not taken
            if self.hist_table[index] > 0:
                self.hist_table[index] -= 1
        
        # Update global history
        self.hist_vector = ((self.hist_vector << 1) | result) & ((1 << global_hist_length) - 1)