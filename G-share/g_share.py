global_hist_length = 14
counter_bits = 2
hist_table_size = 16384


class GShare:
    def __init__(self):
        self.hist_table = [0] * hist_table_size
        self.hist_vector = 0 # Will have to manually make sure this is less than num bits set in global_hist_length

    def predict_branch(self, address): #output will be normalized from 1 to -1
        value = self.hist_table[testing_hash(address)]
        return (value - 2**(counter_bits - 1) + 1 if value >= 2**(counter_bits - 1) else 0) / 2**(counter_bits - 1)

    def update_predictor (self, address, result): # result will be 1 (taken) or 0 (not taken)
        hash_address = testing_hash(address)
        self.hist_table[hash_address] = (self.hist_table[hash_address] + result)
        if (self.hist_table[hash_address] == 2**counter_bits):
            self.hist_table[hash_address] = 2**counter_bits - 1
        elif (result == 0): #needs to be here since not really n bits, may be more efficient way
            self.hist_table[hash_address] -= 1
            if (self.hist_table[hash_address]  < 0):
                self.hist_table[hash_address] = 0
        self.hist_vector = ((self.hist_vector * 2)% 2**global_hist_length) + result



#TESTING FUNCTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def testing_hash (address): 
    return address % hist_table_size
#TESTING FUNCTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#TESTING CODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
gshare = GShare()
print("Testing functions: history")
gshare.update_predictor(22, 1)
gshare.update_predictor(22, 1)
gshare.update_predictor(22, 1)
print("Following should be 7 (assssuming more than 2 bits of history): ", gshare.hist_vector)

print("Going to max history:")
for i in range(global_hist_length - 3):
    gshare.update_predictor(22, 1)
print("Following should be ", 2**global_hist_length - 1, ": ", gshare.hist_vector)

print("Trying to go beyond:")
gshare.update_predictor(22, 1)
print("Following should be ", 2**global_hist_length - 1, ": ", gshare.hist_vector)

print("Testing logic:")
gshare.update_predictor(22, 0)
gshare.update_predictor(22, 1)

print("Following should be ", (2**global_hist_length - 1) - 2, ": ", gshare.hist_vector)



print("Testing functions: predictor output")
print(gshare.predict_branch(23))

print("Testing lower bound")
gshare.update_predictor(23, 0)
print(gshare.predict_branch(23))

print("going to top")
for i in range(2**(counter_bits) - 1):
    gshare.update_predictor(23, 1)
    print(gshare.predict_branch(23))

print("Testing upper bound")
gshare.update_predictor(23, 1)
print(gshare.predict_branch(23))


print("going to bottom")
for i in range(2**(counter_bits) - 1):
    gshare.update_predictor(23, 0)
    print(gshare.predict_branch(23))


print("testing other address real quick (incrementing prev then incrimenting other)")
gshare.update_predictor(23, 1)
print(gshare.predict_branch(23))
print(gshare.predict_branch(24))
#TESTING CODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
