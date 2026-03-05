global_hist_length = 14
counter_bits = 2
hist_table_size = 16384


hist_table = [0] * hist_table_size
hist_vector = 0 # Will have to manually make sure this is less than num bits set in global_hist_length




#TESTING FUNCTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def testing_hash (address): 
    return address % hist_table_size
#TESTING FUNCTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def predict_branch (address): #output will be normalized from 1 to -1
    global hist_table
    global hist_vector

    value = hist_table[testing_hash(address)]
    temp = 0
    if(value >= 2**(counter_bits - 1)):
        temp = 1

    return ((value ) - 2**(counter_bits - 1) + temp) / 2**(counter_bits - 1)


def update_predictor (address, result): # result will be 1 (taken) or 0 (not taken)
    global hist_table
    global hist_vector

    hash_address = testing_hash(address)
    hist_table[hash_address] = (hist_table[hash_address] + result)

    if (hist_table[hash_address] == 2**counter_bits):
        hist_table[hash_address] = 2**counter_bits - 1
    elif (result == 0): #needs to be here since not really n bits, may be more efficient way
        hist_table[hash_address] -= 1
        if (hist_table[hash_address]  < 0):
            hist_table[hash_address] = 0

    hist_vector = ((hist_vector * 2)% 2**global_hist_length) + result




#TESTING CODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print("Testing functions: history")
update_predictor(22, 1)
update_predictor(22, 1)
update_predictor(22, 1)
print("Following should be 7 (assssuming more than 2 bits of history): ", hist_vector)

print("Going to max history:")
for i in range(global_hist_length - 3):
    update_predictor(22, 1)
print("Following should be ", 2**global_hist_length - 1, ": ", hist_vector)

print("Trying to go beyond:")
update_predictor(22, 1)
print("Following should be ", 2**global_hist_length - 1, ": ", hist_vector)

print("Testing logic:")
update_predictor(22, 0)
update_predictor(22, 1)

print("Following should be ", (2**global_hist_length - 1) - 2, ": ", hist_vector)



print("Testing functions: predictor output")
print(predict_branch(23))

print("Testing lower bound")
update_predictor(23, 0)
print(predict_branch(23))

print("going to top")
for i in range(2**(counter_bits) - 1):
    update_predictor(23, 1)
    print(predict_branch(23))

print("Testing upper bound")
update_predictor(23, 1)
print(predict_branch(23))


print("going to bottom")
for i in range(2**(counter_bits) - 1):
    update_predictor(23, 0)
    print(predict_branch(23))


print("testing other address real quick (incrementing prev then incrimenting other)")
update_predictor(23, 1)
print(predict_branch(23))
print(predict_branch(24))
#TESTING CODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!