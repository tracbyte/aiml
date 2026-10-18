def xor(a, b): 
    # Perform XOR operation on bits starting from index 1
    result = []
    for i in range(1, len(b)):
        if a[i] == b[i]:
            result.append('0')
        else:
            result.append('1')
    return ''.join(result) 

def mod2div(dividend, divisor): 
    pick = len(divisor) 
    tmp = dividend[:pick] 
    
    while pick < len(dividend): 
        if tmp[0] == '1':
            tmp = xor(divisor, tmp) + dividend[pick]
        else:
            tmp = xor('0' * pick, tmp) + dividend[pick]
        pick += 1 
    
    if tmp[0] == '1':
        tmp = xor(divisor, tmp)
    else:
        tmp = xor('0' * pick, tmp)
        
    return tmp 

print("--- CRC Sender Side ---") 
data = input("Enter the binary data to send (e.g., 100100): ") 
divisor = input("Enter the Generator Polynomial/Divisor (e.g., 1101): ")

appended_data = data + '0' * (len(divisor) - 1) 
crc_remainder = mod2div(appended_data, divisor) 
transmitted_data = data + crc_remainder 

print(f"\nAppended Data: {appended_data}") 
print(f"CRC Remainder: {crc_remainder}") 
print(f"Transmitted Data: {transmitted_data}") 

print("\n--- CRC Receiver Side ---") 
received_input = input(f"Enter received data (Press Enter to use '{transmitted_data}'): ") 
received_data = received_input if received_input != "" else transmitted_data

receiver_remainder = mod2div(received_data, divisor) 
print(f"\nReceiver Remainder: {receiver_remainder}") 

if '1' in receiver_remainder: 
    print("Status: ERROR DETECTED! The data was corrupted.") 
else: 
    print("Status: SUCCESS! No error detected in transmission.")
